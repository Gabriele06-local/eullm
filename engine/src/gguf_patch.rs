//! Minimal GGUF metadata patcher for Ollama compatibility.
//!
//! Some Ollama GGUF files contain metadata arrays with fewer elements than
//! upstream llama.cpp expects.  For example, `qwen35.rope.dimension_sections`
//! has 3 elements in the Ollama blob but llama.cpp requires 4.
//!
//! This module detects and patches such files during import by inserting
//! zero-filled elements so that the GGUF loads correctly in EULLM's
//! llama.cpp backend.
//!
//! The patcher works at the binary level: it streams through the source
//! file, applies byte-level patches in the metadata section, recalculates
//! alignment padding, and streams the (unchanged) tensor data to the
//! destination.

use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" in little-endian

/// Alignment of the tensor data when the file does not say otherwise.
///
/// The same `GGUF_DEFAULT_ALIGNMENT` ggml uses, and a default rather than a
/// rule: a file may set `general.alignment` to any other power of two, and
/// [`declared_alignment`] reads it.
const DEFAULT_ALIGNMENT: u64 = 32;

/// The metadata key a file uses to set its own tensor-data alignment.
const KEY_ALIGNMENT: &str = "general.alignment";

// GGUF value type IDs.
const TYPE_UINT8: u32 = 0;
const TYPE_INT8: u32 = 1;
const TYPE_UINT16: u32 = 2;
const TYPE_INT16: u32 = 3;
const TYPE_UINT32: u32 = 4;
const TYPE_INT32: u32 = 5;
const TYPE_FLOAT32: u32 = 6;
const TYPE_BOOL: u32 = 7;
const TYPE_STRING: u32 = 8;
const TYPE_ARRAY: u32 = 9;
const TYPE_UINT64: u32 = 10;
const TYPE_INT64: u32 = 11;
const TYPE_FLOAT64: u32 = 12;

/// Size in bytes of a scalar GGUF value type.
///
/// An unrecognised tag is refused rather than measured as zero. Zero was not
/// a size, it was "no idea", and both array paths spent it as a size: they
/// skipped nothing and carried on reading from a position that could not be
/// right, turning one unknown byte into a whole misread header. `fit.rs`
/// stops on a tag it does not know (`Cursor::skip_scalar` returns `None`);
/// this is the same rule, in the shape this module errors in.
fn scalar_size(t: u32) -> io::Result<u64> {
    match t {
        TYPE_UINT8 | TYPE_INT8 | TYPE_BOOL => Ok(1),
        TYPE_UINT16 | TYPE_INT16 => Ok(2),
        TYPE_UINT32 | TYPE_INT32 | TYPE_FLOAT32 => Ok(4),
        TYPE_UINT64 | TYPE_INT64 | TYPE_FLOAT64 => Ok(8),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown GGUF scalar type {t}"),
        )),
    }
}

/// Accept a `general.alignment` a file declares, on ggml's terms.
///
/// `gguf.cpp` refuses the whole file for a value that is zero or not a power
/// of two, and refuses it again if the key is not a `uint32`. Both rules are
/// mirrored here rather than restated: a file ggml will not open is not one
/// to patch, and the alternative — carrying on with 32 because the declared
/// value looked wrong — writes the tensor data at an offset the file does not
/// claim, which is the failure this whole function exists to avoid.
///
/// Zero matters twice over: `align_up` divides by it.
fn accept_alignment(a: u32) -> io::Result<u64> {
    if a == 0 || !a.is_power_of_two() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{KEY_ALIGNMENT} is {a}, which is not a power of two"),
        ));
    }
    Ok(u64::from(a))
}

/// Bytes between the cursor and the end of the file.
///
/// `fit.rs` reads its header into a slice, so every advance there is bounded
/// by `data.len()` for free. This module streams the file instead — it has to
/// handle several GB of tensor data — and gets no such bound, so the one
/// place a length from the file decides how much memory or time to spend has
/// to ask for it. Three seeks, once per string, against a copy that moves
/// gigabytes: not worth measuring.
fn bytes_left(r: &mut (impl Read + Seek)) -> io::Result<u64> {
    let pos = r.stream_position()?;
    let end = r.seek(SeekFrom::End(0))?;
    r.seek(SeekFrom::Start(pos))?;
    Ok(end.saturating_sub(pos))
}

/// Refuse a string array whose count cannot fit in what is left of the file.
///
/// Each element carries an 8-byte length prefix at minimum, so `count`
/// elements need `count * 8` bytes however empty the strings are. Without
/// this the loop below walks a corrupt count one element at a time until it
/// runs into the end of the file: on a multi-GB import that is hundreds of
/// millions of read-and-seek pairs before it gives up, which is a hang to
/// whoever is watching rather than the error it actually is.
fn check_string_array_fits(r: &mut (impl Read + Seek), count: u64) -> io::Result<()> {
    let left = bytes_left(r)?;
    let needed = array_byte_size(count, 8)?;
    if needed > left {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("string array of {count} elements needs at least {needed} bytes, {left} left"),
        ));
    }
    Ok(())
}

/// Byte size of `count` fixed-size elements, refusing overflow.
///
/// Both array-skipping paths multiply an untrusted count by an element size;
/// in release a wrap would skip a small distance and misalign the whole
/// scan, in debug it panics.
fn array_byte_size(count: u64, elem_size: u64) -> io::Result<u64> {
    count.checked_mul(elem_size).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("array byte size overflows: {count} elements of {elem_size} bytes"),
        )
    })
}

/// A patch to apply: extend an array from `current_count` to `target_count`
/// by appending zero-filled elements.
struct ArrayPatch {
    /// File offset of the 8-byte little-endian count field.
    count_offset: u64,
    current_count: u64,
    target_count: u64,
    /// Size of each element in bytes.
    elem_size: u64,
}

/// Known GGUF metadata keys that Ollama may write with too few array elements.
/// Each entry is (key_name, expected_element_count).
const KNOWN_FIXES: &[(&str, u64)] = &[("qwen35.rope.dimension_sections", 4)];

/// Check whether a GGUF file needs Ollama compatibility patches.  If so,
/// write a patched copy to `dst` and return `Ok(true)`.  If no patching is
/// needed, return `Ok(false)` without creating `dst` (the caller should do
/// a normal copy).
///
/// The patch inserts zero-filled array elements where the source has fewer
/// elements than llama.cpp expects, then recalculates the alignment padding
/// before the tensor data section.  Tensor data itself is streamed unchanged.
pub fn patch_gguf_if_needed(src: &Path, dst: &Path) -> io::Result<bool> {
    let mut f = std::fs::File::open(src)?;

    // ── Parse GGUF header ────────────────────────────────────────────
    let magic = read_u32(&mut f)?;
    if magic != GGUF_MAGIC {
        return Ok(false);
    }
    let _version = read_u32(&mut f)?;
    let tensor_count = read_u64(&mut f)?;
    let kv_count = read_u64(&mut f)?;

    // ── Scan metadata KV entries ─────────────────────────────────────
    let mut patches: Vec<ArrayPatch> = Vec::new();
    // Where the tensor data begins is computed from this, twice, so a file
    // that sets its own alignment and is read with 32 gets a patched copy
    // whose tensor data is at an offset nothing in the file agrees with —
    // and the copy is written, not rejected. Read it while passing over it.
    let mut alignment = DEFAULT_ALIGNMENT;

    for _ in 0..kv_count {
        let key = read_gguf_string(&mut f)?;
        let vtype = read_u32(&mut f)?;

        if key == KEY_ALIGNMENT {
            // Checked before the value's type is dispatched on, the way ggml
            // does it: the key is wrong for the file whatever type it turns
            // out to hold, and an array here would otherwise be read as an
            // ordinary array and leave the alignment silently at 32.
            if vtype != TYPE_UINT32 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("{KEY_ALIGNMENT} must be a uint32, found type {vtype}"),
                ));
            }
            alignment = accept_alignment(read_u32(&mut f)?)?;
        } else if vtype == TYPE_ARRAY {
            let elem_type = read_u32(&mut f)?;
            let count_offset = f.stream_position()?;
            let count = read_u64(&mut f)?;
            let elem_sz = if elem_type == TYPE_STRING {
                // String arrays: skip each string individually
                check_string_array_fits(&mut f, count)?;
                for _ in 0..count {
                    skip_gguf_value(&mut f, TYPE_STRING)?;
                }
                0 // not a fixed-size element
            } else {
                let sz = scalar_size(elem_type)?;
                skip_n(&mut f, array_byte_size(count, sz)?)?;
                sz
            };

            // Check if this key needs fixing
            if let Some(&(_, target)) = KNOWN_FIXES.iter().find(|&&(k, _)| k == key.as_str())
                && count < target
                && elem_sz > 0
            {
                patches.push(ArrayPatch {
                    count_offset,
                    current_count: count,
                    target_count: target,
                    elem_size: elem_sz,
                });
                tracing::info!("GGUF patch: {key} has {count} elements, extending to {target}");
            }
        } else {
            skip_gguf_value(&mut f, vtype)?;
        }
    }

    if patches.is_empty() {
        return Ok(false);
    }

    // ── Parse tensor info to find the header/tensor-data boundary ─────
    for _ in 0..tensor_count {
        let _name = read_gguf_string(&mut f)?;
        let n_dims = read_u32(&mut f)?;
        skip_n(&mut f, n_dims as u64 * 8)?; // dimension sizes
        skip_n(&mut f, 4)?; // tensor type
        skip_n(&mut f, 8)?; // data offset (relative to tensor data start)
    }

    let end_of_header = f.stream_position()?;
    let orig_data_start = align_up(end_of_header, alignment);

    // A declared alignment can put the tensor data past the end of the file it
    // came from. Reading on would skip to nowhere and stream an empty rest,
    // producing a truncated copy that looks like a success.
    let src_len = f.metadata()?.len();
    if orig_data_start > src_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "alignment {alignment} puts the tensor data at {orig_data_start}, past the end of a {src_len} byte file"
            ),
        ));
    }

    // Total extra bytes we are inserting into the metadata section.
    let extra_bytes: u64 = patches
        .iter()
        .map(|p| (p.target_count - p.current_count) * p.elem_size)
        .sum();

    let new_data_start = align_up(end_of_header + extra_bytes, alignment);

    // ── Write the patched file ───────────────────────────────────────
    f.seek(SeekFrom::Start(0))?;

    let out = std::fs::File::create(dst)?;
    let mut w = io::BufWriter::with_capacity(8 * 1024 * 1024, out);

    // Sort patches by file offset (they should already be in order, but
    // let's be safe).
    patches.sort_by_key(|p| p.count_offset);

    let mut src_pos: u64 = 0;

    for patch in &patches {
        // Copy everything from current position up to the count field.
        copy_exact(&mut f, &mut w, patch.count_offset - src_pos)?;
        src_pos = patch.count_offset;

        // Read old count (8 bytes), write new count.
        let _old = read_u64(&mut f)?;
        src_pos += 8;
        w.write_all(&patch.target_count.to_le_bytes())?;

        // Copy existing elements.
        let existing = patch.current_count * patch.elem_size;
        copy_exact(&mut f, &mut w, existing)?;
        src_pos += existing;

        // Append zero-filled extra elements.
        let extra = (patch.target_count - patch.current_count) * patch.elem_size;
        let zeros = vec![0u8; extra as usize];
        w.write_all(&zeros)?;
    }

    // Copy remaining header + tensor info up to original padding.
    copy_exact(&mut f, &mut w, end_of_header - src_pos)?;

    // Write new alignment padding.
    // Written in blocks, not allocated whole: the padding used to be under 32
    // bytes because the alignment was a constant 32, and it is now whatever
    // the file asked for, up to a `uint32`.
    write_zeros(&mut w, new_data_start - (end_of_header + extra_bytes))?;

    // Skip old alignment padding in source.
    let old_padding = orig_data_start - end_of_header;
    skip_n(&mut f, old_padding)?;

    // Stream tensor data (the bulk of the file — may be several GB).
    io::copy(&mut f, &mut w)?;

    w.flush()?;
    Ok(true)
}

// ── I/O helpers ──────────────────────────────────────────────────────────

fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

/// Read a GGUF string: a u64 length, then that many raw bytes.
///
/// The length is checked against what is left of the file before anything is
/// allocated, and that ordering is the whole point: `vec![0u8; len]` on a
/// corrupt length does not return an error. Measured, rather than assumed —
/// a length above `isize::MAX` panics with "capacity overflow", which ends
/// `eullm import` with a stack trace instead of the warning and plain copy
/// the caller is written to fall back to. Below that the allocation is not a
/// wall either: `4_611_686_018_427_387_904` bytes was granted on the machine
/// this was written on, because `calloc` hands back lazily-zeroed pages that
/// cost nothing until touched, and the failure only surfaced on the read.
/// Where that laziness is not available — a memory cgroup, a platform
/// without overcommit — the same call aborts the process outright.
///
/// Three outcomes for one corrupt field, none of them an error the caller can
/// act on, so the length is judged before it is spent.
fn read_gguf_string(r: &mut (impl Read + Seek)) -> io::Result<String> {
    let len = read_u64(r)?;
    let left = bytes_left(r)?;
    if len > left {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("string of {len} bytes with {left} left in the file"),
        ));
    }
    // Reachable only on a 32-bit target, where a file can be larger than the
    // address space it is read into.
    let len = usize::try_from(len).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("string of {len} bytes does not fit in memory"),
        )
    })?;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

/// Advance the reader by `n` bytes (seek if possible).
///
/// `n as i64` reads as a widening no-op and is not one: every value above
/// `i64::MAX` comes out negative, so a forward skip becomes a backward one.
/// `array_byte_size` catches the multiplication wrapping u64, and `2^61`
/// elements of 4 bytes survives it — `2^63` is a fine u64 and a negative i64.
///
/// This is not a bug being fixed, and saying otherwise would be easy: the
/// seek already refuses, because no real file sits far enough in for
/// `position - 2^63` to land at or above zero. What it refuses with is
/// "invalid seek to a negative or overflowing position", which is a true
/// sentence about a cast nobody reading the call site knows happened, and a
/// misleading one about a skip that was asked for forwards. The check states
/// the limit where the limit is, rather than leaving the arithmetic to be
/// caught by something downstream that does not know what it is catching.
fn skip_n(r: &mut (impl Read + Seek), n: u64) -> io::Result<()> {
    let n = i64::try_from(n).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("skip of {n} bytes is past any position a file can have"),
        )
    })?;
    r.seek(SeekFrom::Current(n))?;
    Ok(())
}

/// Skip a single GGUF value in the stream (used during header scanning).
fn skip_gguf_value(r: &mut (impl Read + Seek), vtype: u32) -> io::Result<()> {
    match vtype {
        TYPE_UINT8 | TYPE_INT8 | TYPE_BOOL => skip_n(r, 1),
        TYPE_UINT16 | TYPE_INT16 => skip_n(r, 2),
        TYPE_UINT32 | TYPE_INT32 | TYPE_FLOAT32 => skip_n(r, 4),
        TYPE_UINT64 | TYPE_INT64 | TYPE_FLOAT64 => skip_n(r, 8),
        TYPE_STRING => {
            let len = read_u64(r)?;
            skip_n(r, len)
        }
        TYPE_ARRAY => {
            let elem_type = read_u32(r)?;
            let count = read_u64(r)?;
            if elem_type == TYPE_STRING {
                check_string_array_fits(r, count)?;
                for _ in 0..count {
                    skip_gguf_value(r, TYPE_STRING)?;
                }
                Ok(())
            } else {
                skip_n(r, array_byte_size(count, scalar_size(elem_type)?)?)?;
                Ok(())
            }
        }
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown GGUF value type {vtype}"),
        )),
    }
}

/// Write `n` zero bytes using an 8 KB buffer.
fn write_zeros(w: &mut impl Write, mut n: u64) -> io::Result<()> {
    let buf = [0u8; 8192];
    while n > 0 {
        let chunk = n.min(buf.len() as u64) as usize;
        w.write_all(&buf[..chunk])?;
        n -= chunk as u64;
    }
    Ok(())
}

/// Copy exactly `n` bytes from reader to writer using an 8 KB buffer.
fn copy_exact(r: &mut impl Read, w: &mut impl Write, mut n: u64) -> io::Result<()> {
    let mut buf = [0u8; 8192];
    while n > 0 {
        let chunk = n.min(buf.len() as u64) as usize;
        r.read_exact(&mut buf[..chunk])?;
        w.write_all(&buf[..chunk])?;
        n -= chunk as u64;
    }
    Ok(())
}

/// Round `v` up to the next multiple of `alignment`, which must be non-zero
/// (`accept_alignment` is the only thing that produces one). The multiply
/// cannot overflow for any value this module passes: `v` is a position in a
/// file and the result is below `v + alignment`.
fn align_up(v: u64, alignment: u64) -> u64 {
    v.div_ceil(alignment) * alignment
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Build a whole GGUF: one short `qwen35.rope.dimension_sections`, one
    /// tensor, and optionally a declared `general.alignment`. Returns the
    /// bytes and the offset the metadata ends at, which is what both the
    /// source and the patched copy pad from.
    fn gguf_with(alignment: Option<u32>, sections: &[u32], data: &[u8]) -> (Vec<u8>, u64) {
        fn put_str(b: &mut Vec<u8>, v: &str) {
            b.extend_from_slice(&(v.len() as u64).to_le_bytes());
            b.extend_from_slice(v.as_bytes());
        }

        let mut b = Vec::new();
        b.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes()); // version
        b.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
        b.extend_from_slice(&(1 + u64::from(alignment.is_some())).to_le_bytes());

        if let Some(a) = alignment {
            put_str(&mut b, KEY_ALIGNMENT);
            b.extend_from_slice(&TYPE_UINT32.to_le_bytes());
            b.extend_from_slice(&a.to_le_bytes());
        }

        put_str(&mut b, "qwen35.rope.dimension_sections");
        b.extend_from_slice(&TYPE_ARRAY.to_le_bytes());
        b.extend_from_slice(&TYPE_UINT32.to_le_bytes());
        b.extend_from_slice(&(sections.len() as u64).to_le_bytes());
        for v in sections {
            b.extend_from_slice(&v.to_le_bytes());
        }

        // One tensor info: name, n_dims, one dimension, type, offset. The
        // name's length is load-bearing — see the alignment test, which needs
        // the metadata to end at a specific offset to mean anything.
        put_str(&mut b, "ten");
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&(data.len() as u64).to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());

        let end_of_header = b.len() as u64;
        let data_start = align_up(
            end_of_header,
            alignment.map_or(DEFAULT_ALIGNMENT, u64::from),
        );
        b.resize(data_start as usize, 0);
        b.extend_from_slice(data);
        (b, end_of_header)
    }

    fn temp_pair(what: &str) -> (std::path::PathBuf, std::path::PathBuf) {
        let dir = std::env::temp_dir();
        let id = uuid::Uuid::new_v4();
        (
            dir.join(format!("eullm-{what}-src-{id}.gguf")),
            dir.join(format!("eullm-{what}-dst-{id}.gguf")),
        )
    }

    /// Run the patcher over a built file and hand back the patched bytes.
    fn patch(bytes: &[u8], what: &str) -> io::Result<Option<Vec<u8>>> {
        let (src, dst) = temp_pair(what);
        std::fs::write(&src, bytes)?;
        let r = patch_gguf_if_needed(&src, &dst);
        let out = match &r {
            Ok(true) => Some(std::fs::read(&dst)?),
            _ => None,
        };
        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(&dst);
        r.map(|_| out)
    }

    /// The offset the tensor data lands at is computed from the alignment,
    /// twice, and the alignment used to be the constant 32 whatever the file
    /// said.
    ///
    /// Which offset that produces is not simply "32 instead of 64": the old
    /// code also streamed the source's own padding through unread, so for
    /// most sizes the two errors cancelled and the data landed where it
    /// belonged anyway. The first fixture written here was one of those, and
    /// passed against the constant it was meant to catch. It is the *growth*
    /// in padding that has to differ, and at 158 bytes of metadata it does:
    /// the four bytes the patch appends cross a 32-boundary but not a
    /// 64-boundary, which puts the data at 224 under the old constant and 192
    /// under the declared alignment. Checked in both directions.
    #[test]
    fn a_declared_alignment_decides_where_the_tensor_data_goes() {
        let data = b"TENSOR-BYTES";
        let (bytes, end_of_header) = gguf_with(Some(64), &[1, 2, 3], data);
        assert_eq!(
            end_of_header, 158,
            "the fixture stopped landing where 32 and 64 disagree"
        );
        // Where the data must end up. The old constant put it at 224: the
        // padding it wrote grew by 32 and the source's own 34 bytes of
        // padding were then copied through on top.
        let patched_start = align_up(end_of_header + 4, 64);
        assert_eq!(patched_start, 192);

        let out = patch(&bytes, "align64").unwrap().expect("should patch");

        // One u32 element appended, so the metadata is four bytes longer.
        assert_eq!(
            &out[patched_start as usize..],
            data,
            "tensor data is not where the declared alignment puts it"
        );
        assert!(
            out[(end_of_header + 4) as usize..patched_start as usize]
                .iter()
                .all(|&b| b == 0),
            "the gap before it must be padding"
        );
    }

    /// The default path, unchanged: no key, so 32, which is what every file
    /// anyone actually has says by saying nothing.
    #[test]
    fn a_file_that_declares_no_alignment_still_pads_to_32() {
        let data = b"TENSOR-BYTES";
        let (bytes, end_of_header) = gguf_with(None, &[1, 2, 3], data);
        let out = patch(&bytes, "align-default")
            .unwrap()
            .expect("should patch");
        let patched_start = align_up(end_of_header + 4, DEFAULT_ALIGNMENT);
        assert_eq!(&out[patched_start as usize..], data);
    }

    /// ggml refuses the file outright for these, so patching it is pointless
    /// and guessing 32 would write the data where the file does not claim it
    /// is. Zero would also divide by zero in `align_up`.
    #[test]
    fn an_alignment_that_is_not_a_power_of_two_is_refused() {
        for good in [1u32, 2, 32, 64, 4096, 1 << 31] {
            assert_eq!(accept_alignment(good).unwrap(), u64::from(good));
        }
        for bad in [0u32, 3, 48, 100, u32::MAX] {
            assert!(accept_alignment(bad).is_err(), "{bad} must be refused");
        }

        let (bytes, _) = gguf_with(Some(48), &[1, 2, 3], b"x");
        assert!(patch(&bytes, "align48").is_err());
    }

    /// A declared alignment can put the tensor data past the end of the file
    /// that declared it. Skipping to nowhere and streaming the empty rest
    /// writes a truncated copy and calls it a success.
    #[test]
    fn an_alignment_past_the_end_of_the_file_is_refused() {
        // Built well-formed, then cut short: a file declaring a megabyte of
        // alignment without carrying a megabyte of padding. `gguf_with` pads
        // to what it declares, so truncating is what makes this the malformed
        // case rather than a large well-formed one.
        let (mut bytes, end_of_header) = gguf_with(Some(1 << 20), &[1, 2, 3], b"x");
        bytes.truncate(end_of_header as usize + 8);
        assert!(
            (bytes.len() as u64) < (1 << 20),
            "the fixture must be shorter than the alignment it declares"
        );
        assert!(patch(&bytes, "align-past-end").is_err());
    }

    /// An array value's raw bytes: element type followed by count.
    fn array_value(elem_type: u32, count: u64) -> Cursor<Vec<u8>> {
        let mut b = Vec::new();
        b.extend_from_slice(&elem_type.to_le_bytes());
        b.extend_from_slice(&count.to_le_bytes());
        Cursor::new(b)
    }

    /// A count whose byte size overflows u64 must be refused, not wrapped
    /// into a small skip that misaligns the whole scan (release) or panics
    /// (debug).
    #[test]
    fn overflowing_array_byte_size_is_rejected() {
        assert!(array_byte_size(u64::MAX, 4).is_err());
        let mut c = array_value(TYPE_UINT32, u64::MAX);
        assert!(skip_gguf_value(&mut c, TYPE_ARRAY).is_err());
    }

    /// A length from the file decides how big an allocation is, and none of
    /// the ways that goes wrong is an error the caller can act on: past
    /// `isize::MAX` it panics on capacity overflow and takes `eullm import`
    /// with it; under it, an allocator with overcommit grants exabytes that
    /// cost nothing until touched, and one without aborts. So the length is
    /// checked before it is spent, not after.
    #[test]
    fn a_string_longer_than_the_file_is_refused_before_allocating() {
        let mut c = Cursor::new(u64::MAX.to_le_bytes().to_vec());
        assert!(read_gguf_string(&mut c).is_err());

        // One byte short is still refused: the bound is what is left, not a
        // round number someone picked.
        let mut b = 4u64.to_le_bytes().to_vec();
        b.extend_from_slice(b"abc");
        assert!(read_gguf_string(&mut Cursor::new(b)).is_err());
    }

    #[test]
    fn an_ordinary_string_still_reads() {
        let mut b = 5u64.to_le_bytes().to_vec();
        b.extend_from_slice(b"qwen3");
        let mut c = Cursor::new(b);
        assert_eq!(read_gguf_string(&mut c).unwrap(), "qwen3");
        assert_eq!(c.position(), 8 + 5);
    }

    /// Zero was never a size. Measuring an unknown tag as zero skipped nothing
    /// and left the scan reading a value from the middle of the previous one.
    #[test]
    fn an_unknown_array_element_type_stops_the_scan() {
        assert!(scalar_size(99).is_err());
        let mut c = array_value(99, 1);
        c.get_mut().extend_from_slice(&[0u8; 64]);
        assert!(skip_gguf_value(&mut c, TYPE_ARRAY).is_err());
    }

    /// Survives the overflow check and still breaks the seek: 2^61 elements of
    /// 4 bytes is 2^63, which is a perfectly good u64 and a negative i64.
    ///
    /// Not a regression test, and it would be dishonest to file it as one:
    /// the cast this replaces failed here too, by seeking to a position no
    /// file has. What is pinned is that the refusal is the one this module
    /// decided on, at the point where the size stops being representable.
    #[test]
    fn a_size_that_fits_u64_but_not_a_seek_is_refused() {
        let n = 1u64 << 61;
        assert_eq!(array_byte_size(n, 4).unwrap(), 1u64 << 63);
        assert!(skip_n(&mut Cursor::new(vec![0u8; 16]), 1u64 << 63).is_err());
        assert!(skip_gguf_value(&mut array_value(TYPE_UINT32, n), TYPE_ARRAY).is_err());
    }

    /// Each element carries an 8-byte length prefix, so a count the file
    /// cannot hold is refused up front instead of being walked one element at
    /// a time until the reads run out — which on a multi-GB file is a hang.
    #[test]
    fn a_string_array_longer_than_the_file_is_refused_at_once() {
        // A million elements need 8 MB of length prefixes alone, against 64
        // bytes of file. Small enough that `count * 8` does not overflow, so
        // this reaches the size check rather than stopping one step earlier.
        let mut c = array_value(TYPE_STRING, 1_000_000);
        c.get_mut().extend_from_slice(&[0u8; 64]);
        assert!(check_string_array_fits(&mut c, 1_000_000).is_err());
        assert!(skip_gguf_value(&mut c, TYPE_ARRAY).is_err());

        // A count large enough to overflow the multiplication stops there
        // instead, which is the same refusal by the earlier of the two gates.
        assert!(check_string_array_fits(&mut Cursor::new(vec![0u8; 64]), u64::MAX).is_err());
    }

    #[test]
    fn ordinary_array_values_still_skip() {
        assert_eq!(array_byte_size(3, 4).unwrap(), 12);
        let mut c = array_value(TYPE_UINT32, 3);
        // 3 elements of 4 bytes of payload must follow the header.
        c.get_mut().extend_from_slice(&[0u8; 12]);
        assert!(skip_gguf_value(&mut c, TYPE_ARRAY).is_ok());
        assert_eq!(c.position(), 4 + 8 + 12);
    }
}
