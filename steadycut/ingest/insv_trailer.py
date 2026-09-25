"""Read the metadata trailer appended to Insta360 .insv/.insp files.

The trailer sits at the end of the file and is walked *backwards* from EOF:
the last 32 bytes are a magic string, preceded by a version field and the
total trailer size.  Each record is announced by a 6-byte header (2-byte id,
4-byte size, little endian) and the payload occupies the `size` bytes that
precede that header.

X4/X5 do not pack the records contiguously -- runs of zero padding of
non-constant length separate them -- so walking record-to-record the way an
X3 parser does desynchronises.  Reading the index that those models write
between the magic number and the first record is the reliable route, and the
walker here tolerates padding so it also works on older layouts.

Nothing outside the trailer is read, so this is safe to run against a file
that is still being copied *provided* the copy has finished writing the tail;
callers should check that the size has settled first.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

# The magic is stored as 32 ASCII characters, not as 16 decoded bytes -- the
# file literally ends with the text "8db42d69...e026bf".
MAGIC = b"8db42d694ccc418790edff439fe026bf"

# Record ids seen in the wild.  0x0300 (IMU) and 0x0700 (GPS on X4) are the
# ones this project needs; the rest are identified so an unknown id is
# genuinely unknown rather than merely unlabelled.
RECORD_NAMES = {
    0x0101: "makernotes",
    0x0200: "thumbnail",
    0x0300: "imu",
    0x0400: "exposure",
    0x0500: "unknown_0500",
    0x0600: "timestamps",
    0x0700: "gps",
    0x0900: "unknown_0900",
    0x0A00: "unknown_0a00",
    0x0B00: "unknown_0b00",
}

# timecode (ms) + accel xyz + gyro xyz, all little endian.
IMU_RECORD = struct.Struct("<Q6d")


@dataclass(frozen=True)
class TrailerRecord:
    id: int
    name: str
    offset: int
    size: int


@dataclass(frozen=True)
class Trailer:
    version: int
    size: int
    records: dict[int, TrailerRecord]

    def get(self, record_id: int) -> TrailerRecord | None:
        return self.records.get(record_id)


class NotInsta360Error(ValueError):
    """The file carries no Insta360 trailer magic."""


def read_trailer(path: str | Path) -> Trailer:
    """Locate and index the trailer of an .insv/.insp file."""
    path = Path(path)
    file_size = path.stat().st_size

    with path.open("rb") as fh:
        # Magic occupies the final 32 bytes; version and trailer size are the
        # 8 bytes before it.
        fh.seek(-(len(MAGIC) + 8), 2)
        tail = fh.read(len(MAGIC) + 8)
        # Camera-original X4 files store size first and version second; the
        # version is a small integer (3 on X4) while the size runs to tens of
        # megabytes, which is what tells the two words apart.
        trailer_size, version = struct.unpack("<II", tail[:8])
        if tail[8:] != MAGIC:
            raise NotInsta360Error(f"{path.name}: no Insta360 trailer magic")

        # The recorded size spans the records plus the 32-byte magic, but not
        # the version and size words themselves, so those 8 bytes come off as
        # well to land on the first byte of the trailer.
        trailer_start = file_size - trailer_size - 8
        cursor = file_size - (len(MAGIC) + 8)
        records: dict[int, TrailerRecord] = {}

        while cursor - 6 >= trailer_start:
            fh.seek(cursor - 6)
            record_id, size = struct.unpack("<HI", fh.read(6))

            payload_end = cursor - 6
            payload_start = payload_end - size
            plausible = (
                record_id in RECORD_NAMES
                and size > 0
                and payload_start >= trailer_start
            )

            if not plausible:
                # X4/X5 pad between records with runs of zeroes of varying
                # length, so a header is not always where the previous record
                # ends. Step back a byte and re-test rather than giving up --
                # requiring a known id keeps this from locking onto noise.
                cursor -= 1
                continue

            records[record_id] = TrailerRecord(
                id=record_id,
                name=RECORD_NAMES.get(record_id, f"unknown_{record_id:#06x}"),
                offset=payload_start,
                size=size,
            )
            cursor = payload_start

        return Trailer(version=version, size=trailer_size, records=records)


def read_imu(path: str | Path, trailer: Trailer | None = None) -> list[dict]:
    """Return the IMU samples: timestamp in ms, accel in g, gyro in deg/s.

    Axis naming follows Insta360's own order (pitch, yaw, roll) rather than
    the conventional roll/pitch/yaw, and the mapping to world axes differs
    per model, so callers must apply the model's orientation before using
    these as rotations.
    """
    path = Path(path)
    trailer = trailer or read_trailer(path)
    record = trailer.get(0x0300)
    if record is None:
        return []

    count = record.size // IMU_RECORD.size
    with path.open("rb") as fh:
        fh.seek(record.offset)
        raw = fh.read(count * IMU_RECORD.size)

    samples = []
    for timecode, ax, ay, az, gx, gy, gz in IMU_RECORD.iter_unpack(raw):
        samples.append(
            {
                "t_ms": timecode,
                "accel": (ax, ay, az),
                "gyro": (gx, gy, gz),
            }
        )
    return samples


def _main() -> None:
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("usage: insv_trailer.py <file.insv>")

    path = Path(sys.argv[1])
    trailer = read_trailer(path)
    print(f"file          {path.name}")
    print(f"trailer size  {trailer.size} bytes (version {trailer.version})")
    print(f"records       {len(trailer.records)}")
    for record in sorted(trailer.records.values(), key=lambda r: r.id):
        print(f"  {record.id:#06x}  {record.name:<16} {record.size:>10} bytes")

    samples = read_imu(path, trailer)
    if not samples:
        print("\nno IMU record found")
        return

    span_s = (samples[-1]["t_ms"] - samples[0]["t_ms"]) / 1000.0
    print(f"\nIMU           {len(samples)} samples")
    print(f"  first t     {samples[0]['t_ms']} ms")
    print(f"  last t      {samples[-1]['t_ms']} ms")
    print(f"  span        {span_s:.2f} s")
    if span_s > 0:
        print(f"  rate        {(len(samples) - 1) / span_s:.1f} Hz")
    print(f"  sample[0]   accel={samples[0]['accel']} gyro={samples[0]['gyro']}")


if __name__ == "__main__":
    _main()
