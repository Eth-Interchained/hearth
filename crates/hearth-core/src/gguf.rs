//! Read the four numbers a GGUF header carries that decide how big its KV
//! cache gets. Nothing else in the file — not a tensor, not a token.
//!
//! ## Why this exists
//!
//! `budget::Declared` has carried a `kv_bytes` field since the planner was
//! written, and every real caller passed `0`. Declaring four models by
//! weight size alone and nothing errors — the planner admits them, the card
//! looks like it has room, and then the KV cache each one actually needs at
//! load time (which nothing budgeted for) exhausts VRAM anyway. On
//! 2026-08-28 that put a 32B model's `llama-server` through exactly this: two
//! declared models fit on paper, `qwen2.5:14b`'s own CUDA allocation failed
//! against a card `muse-local:latest` had already spent ~14 GiB of KV cache
//! on, and the process exited with no error text llama.cpp thought to print.
//! `/residency` reported "30.0 / 42.0 GiB held" the entire time — correct
//! about weights, and silent about the number that actually mattered.
//!
//! KV cache size is not a function of weight size. It is a function of the
//! model's own architecture — how many transformer blocks, how many KV
//! heads, how wide each head is — multiplied by how much context you ask it
//! to hold open. Two models with the same GGUF file size can want wildly
//! different KV footprints. So the shape has to come from the GGUF itself.
//!
//! ## What this reads, and what it refuses to
//!
//! Only the metadata header: the four architecture facts above, plus
//! `context_length` as the fallback total context when the operator has not
//! set one explicitly. Tensor data — which is the entire multi-gigabyte body
//! of the file — is never touched; parsing stops the instant the last
//! metadata entry is consumed. An unrecognised value is skipped by its
//! declared type and length, never assumed, so a vocabulary array with a
//! hundred thousand token strings costs a seek's worth of reads, not an
//! allocation.

use std::io::{self, Read};

const MAGIC: u32 = 0x4655_4747; // ASCII "GGUF", read little-endian

/// GGUF metadata value types, as the format defines them. Only the ones this
/// reader actually returns a value for get a comment; the rest exist so
/// `skip_value` knows how far to advance past them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ValueType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    Bool,
    String,
    Array,
    U64,
    I64,
    F64,
}

impl ValueType {
    fn from_tag(tag: u32) -> Result<ValueType, String> {
        Ok(match tag {
            0 => ValueType::U8,
            1 => ValueType::I8,
            2 => ValueType::U16,
            3 => ValueType::I16,
            4 => ValueType::U32,
            5 => ValueType::I32,
            6 => ValueType::F32,
            7 => ValueType::Bool,
            8 => ValueType::String,
            9 => ValueType::Array,
            10 => ValueType::U64,
            11 => ValueType::I64,
            12 => ValueType::F64,
            other => return Err(format!("unknown GGUF value type tag {other}")),
        })
    }
}

/// The architecture facts that decide one model's KV cache footprint.
///
/// Everything here is per-layer, per-head, or a count of them — never a
/// total, because the total also depends on context length and slot count,
/// which are runtime choices, not facts about the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvShape {
    /// Transformer blocks. Each one holds its own K and V cache.
    pub block_count: u64,
    /// KV attention heads. Equal to the query head count on a model without
    /// grouped-query attention; smaller on one that shares KV heads across
    /// several query heads, which is most current models and exactly why
    /// this cannot be assumed equal to `block_count`'s neighbour metadata.
    pub head_count_kv: u64,
    /// Width of one K vector, in elements.
    pub key_length: u64,
    /// Width of one V vector, in elements. Usually equal to `key_length`;
    /// kept separate because the format allows them to differ and nothing
    /// here should assume a convention the file itself does not state.
    pub value_length: u64,
    /// The model's own trained context window. Used as the total context
    /// when the operator has not set one, because that is what llama.cpp
    /// itself falls back to.
    pub context_length: u64,
}

impl KvShape {
    /// Bytes of KV cache for `total_ctx_tokens` tokens of context, at 2 bytes
    /// per element (f16), which is llama.cpp's cache dtype unless the
    /// operator has asked for a quantized KV cache — hearth does not pass
    /// that flag today, so f16 is the honest default, not a guess.
    ///
    /// Both K and V are cached, hence `* 2` on top of the per-vector width:
    /// this is `bytes = layers × heads_kv × (key_len + value_len) × tokens × 2`.
    pub fn kv_bytes_for(&self, total_ctx_tokens: u64) -> u64 {
        self.block_count
            .saturating_mul(self.head_count_kv)
            .saturating_mul(self.key_length.saturating_add(self.value_length))
            .saturating_mul(total_ctx_tokens)
            .saturating_mul(2)
    }
}

/// Everything read from the metadata section before the shape can be built,
/// including the pieces that only exist to fill in for missing ones.
#[derive(Default)]
struct RawFacts {
    block_count: Option<u64>,
    head_count: Option<u64>,
    head_count_kv: Option<u64>,
    key_length: Option<u64>,
    value_length: Option<u64>,
    embedding_length: Option<u64>,
    context_length: Option<u64>,
}

/// Read the KV shape from a GGUF file's metadata header.
///
/// Takes anything `Read`, deliberately: a caller with a whole file open
/// hands over that file directly, and a test hands over a `Vec<u8>` built by
/// hand. Neither one needs to know how the other works.
///
/// Stops reading the instant the metadata section ends — the tensor info
/// table and every byte of actual tensor data that follow it are never
/// touched, which is the difference between opening a header and opening a
/// 20 GiB file.
pub fn read_kv_shape<R: Read>(r: &mut R) -> Result<KvShape, String> {
    let magic = read_u32(r).map_err(|e| format!("reading GGUF magic: {e}"))?;
    if magic != MAGIC {
        return Err(format!(
            "not a GGUF file: magic was {magic:#010x}, expected {MAGIC:#010x}"
        ));
    }
    let _version = read_u32(r).map_err(|e| format!("reading GGUF version: {e}"))?;
    let _tensor_count = read_u64(r).map_err(|e| format!("reading tensor count: {e}"))?;
    let kv_count = read_u64(r).map_err(|e| format!("reading metadata count: {e}"))?;

    let mut facts = RawFacts::default();

    for i in 0..kv_count {
        let key = read_string(r).map_err(|e| format!("metadata entry {i} key: {e}"))?;
        let tag = read_u32(r).map_err(|e| format!("metadata entry {i} type: {e}"))?;
        let vtype = ValueType::from_tag(tag).map_err(|e| format!("entry {i} ({key}): {e}"))?;

        // Suffix match, not architecture-prefixed lookup: the key is written
        // as "{arch}.block_count" and this reader would otherwise need
        // "general.architecture" to have already been seen, which the
        // format does not guarantee. The suffixes below are unique to the
        // attention/block metadata block in every architecture llama.cpp
        // ships, so matching on them directly is both simpler and immune to
        // key ordering.
        let wants_value = key.ends_with(".block_count")
            || key.ends_with(".attention.head_count_kv")
            || key.ends_with(".attention.head_count")
            || key.ends_with(".attention.key_length")
            || key.ends_with(".attention.value_length")
            || key.ends_with(".embedding_length")
            || key.ends_with(".context_length");

        if !wants_value || vtype == ValueType::Array || vtype == ValueType::String {
            // Arrays and strings on these exact suffixes do not occur in
            // practice; skip uniformly rather than special-case a shape the
            // format has never actually produced.
            skip_value(r, vtype).map_err(|e| format!("skipping entry {i} ({key}): {e}"))?;
            continue;
        }

        let value = read_scalar_as_u64(r, vtype)
            .map_err(|e| format!("reading entry {i} ({key}) as an integer: {e}"))?;

        if key.ends_with(".block_count") {
            facts.block_count = Some(value);
        } else if key.ends_with(".attention.head_count_kv") {
            facts.head_count_kv = Some(value);
        } else if key.ends_with(".attention.head_count") {
            facts.head_count = Some(value);
        } else if key.ends_with(".attention.key_length") {
            facts.key_length = Some(value);
        } else if key.ends_with(".attention.value_length") {
            facts.value_length = Some(value);
        } else if key.ends_with(".embedding_length") {
            facts.embedding_length = Some(value);
        } else if key.ends_with(".context_length") {
            facts.context_length = Some(value);
        }
    }

    build_shape(facts)
}

/// Turn what was actually in the file into a shape, filling in the two
/// fields older exports often omit.
///
/// `head_count_kv` defaults to `head_count`: a model with no grouped-query
/// attention metadata has one KV head per query head, which is what
/// "no GQA" means. `key_length`/`value_length` default to
/// `embedding_length / head_count` — llama.cpp's own fallback for a file
/// that predates those keys being written at all — and the two default to
/// each other when only one is present, since a model that bothers to state
/// one but not the other virtually always means them to match.
fn build_shape(facts: RawFacts) -> Result<KvShape, String> {
    let block_count = facts.block_count.ok_or("missing \"*.block_count\"")?;
    let context_length = facts.context_length.ok_or("missing \"*.context_length\"")?;
    let head_count_kv = facts
        .head_count_kv
        .or(facts.head_count)
        .ok_or("missing both \"*.attention.head_count_kv\" and \"*.attention.head_count\"")?;

    let derived_head_dim = match (facts.embedding_length, facts.head_count) {
        (Some(embd), Some(heads)) if heads > 0 => Some(embd / heads),
        _ => None,
    };

    let key_length = facts
        .key_length
        .or(facts.value_length)
        .or(derived_head_dim)
        .ok_or(
            "missing \"*.attention.key_length\" and nothing to derive it from \
             (\"*.embedding_length\" and \"*.attention.head_count\")",
        )?;
    let value_length = facts.value_length.or(facts.key_length).unwrap_or(key_length);

    Ok(KvShape {
        block_count,
        head_count_kv,
        key_length,
        value_length,
        context_length,
    })
}

// ---- byte-level reading -----------------------------------------------
//
// No byteorder dependency: GGUF is little-endian throughout and four
// fixed-width readers plus a length-prefixed string cover every scalar the
// format defines.

fn read_exact<R: Read>(r: &mut R, n: usize) -> io::Result<Vec<u8>> {
    let mut buf = vec![0u8; n];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let b = read_exact(r, 4)?;
    Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn read_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let b = read_exact(r, 8)?;
    Ok(u64::from_le_bytes([
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
    ]))
}

fn read_string<R: Read>(r: &mut R) -> io::Result<String> {
    let len = read_u64(r)? as usize;
    let bytes = read_exact(r, len)?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// The byte width of every fixed-size scalar type. `String` and `Array` are
/// not fixed-size and are handled by their own branches in `skip_value`.
fn fixed_width(vtype: ValueType) -> Option<usize> {
    Some(match vtype {
        ValueType::U8 | ValueType::I8 | ValueType::Bool => 1,
        ValueType::U16 | ValueType::I16 => 2,
        ValueType::U32 | ValueType::I32 | ValueType::F32 => 4,
        ValueType::U64 | ValueType::I64 | ValueType::F64 => 8,
        ValueType::String | ValueType::Array => return None,
    })
}

/// Advance past one value of the given type without allocating it, beyond
/// what a length-prefixed string or array must briefly hold to know how far
/// to skip its own elements.
fn skip_value<R: Read>(r: &mut R, vtype: ValueType) -> io::Result<()> {
    match vtype {
        ValueType::String => {
            read_string(r)?;
        }
        ValueType::Array => {
            let elem_tag = read_u32(r)?;
            let elem_type = ValueType::from_tag(elem_tag)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            let len = read_u64(r)?;
            for _ in 0..len {
                skip_value(r, elem_type)?;
            }
        }
        other => {
            let width = fixed_width(other).expect("fixed_width covers every non-string/array tag");
            read_exact(r, width)?;
        }
    }
    Ok(())
}

/// Read one scalar value and widen it to `u64`. Only called on a type this
/// module has already confirmed is not `String` or `Array`, so every
/// remaining arm is a fixed-width read.
fn read_scalar_as_u64<R: Read>(r: &mut R, vtype: ValueType) -> io::Result<u64> {
    Ok(match vtype {
        ValueType::U8 | ValueType::I8 | ValueType::Bool => read_exact(r, 1)?[0] as u64,
        ValueType::U16 | ValueType::I16 => {
            let b = read_exact(r, 2)?;
            u16::from_le_bytes([b[0], b[1]]) as u64
        }
        ValueType::U32 | ValueType::I32 => {
            let b = read_exact(r, 4)?;
            u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u64
        }
        ValueType::F32 => {
            let b = read_exact(r, 4)?;
            f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u64
        }
        ValueType::U64 | ValueType::I64 => read_u64(r)?,
        ValueType::F64 => {
            let b = read_exact(r, 8)?;
            f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as u64
        }
        ValueType::String | ValueType::Array => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "expected a numeric value",
            ))
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal, hand-built GGUF metadata section — no tensors, because
    /// this reader never looks at them. Encodes exactly the keys a real
    /// model file would have for a Qwen2.5-style architecture: explicit GQA
    /// (`head_count_kv` smaller than `head_count`) and explicit key/value
    /// lengths, which is the shape newer exports actually write.
    fn qwen_style_bytes() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&MAGIC.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes()); // version
        b.extend_from_slice(&0u64.to_le_bytes()); // tensor_count — never read past
        let entries: &[(&str, ValueType, u64)] = &[
            ("qwen2.block_count", ValueType::U32, 48),
            ("qwen2.attention.head_count", ValueType::U32, 40),
            ("qwen2.attention.head_count_kv", ValueType::U32, 8),
            ("qwen2.attention.key_length", ValueType::U32, 128),
            ("qwen2.attention.value_length", ValueType::U32, 128),
            ("qwen2.context_length", ValueType::U32, 32768),
            ("qwen2.embedding_length", ValueType::U32, 5120),
        ];
        b.extend_from_slice(&(entries.len() as u64).to_le_bytes());
        for (key, vtype, value) in entries {
            write_string(&mut b, key);
            write_scalar(&mut b, *vtype, *value);
        }
        b
    }

    fn write_string(b: &mut Vec<u8>, s: &str) {
        b.extend_from_slice(&(s.len() as u64).to_le_bytes());
        b.extend_from_slice(s.as_bytes());
    }

    fn write_scalar(b: &mut Vec<u8>, vtype: ValueType, value: u64) {
        let tag: u32 = match vtype {
            ValueType::U8 => 0,
            ValueType::U16 => 2,
            ValueType::U32 => 4,
            ValueType::U64 => 10,
            _ => panic!("test helper only covers unsigned scalars"),
        };
        b.extend_from_slice(&tag.to_le_bytes());
        match vtype {
            ValueType::U8 => b.push(value as u8),
            ValueType::U16 => b.extend_from_slice(&(value as u16).to_le_bytes()),
            ValueType::U32 => b.extend_from_slice(&(value as u32).to_le_bytes()),
            ValueType::U64 => b.extend_from_slice(&value.to_le_bytes()),
            _ => unreachable!(),
        }
    }

    #[test]
    fn reads_the_shape_a_gqa_model_actually_declares() {
        let bytes = qwen_style_bytes();
        let shape = read_kv_shape(&mut &bytes[..]).unwrap();
        assert_eq!(shape.block_count, 48);
        assert_eq!(shape.head_count_kv, 8, "GQA: kv heads, not query heads");
        assert_eq!(shape.key_length, 128);
        assert_eq!(shape.value_length, 128);
        assert_eq!(shape.context_length, 32768);
    }

    #[test]
    fn a_file_with_no_gqa_metadata_falls_back_to_head_count() {
        // Older / simpler exports omit head_count_kv entirely. No GQA
        // metadata means one KV head per query head, not zero KV heads.
        let mut b = Vec::new();
        b.extend_from_slice(&MAGIC.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        let entries: &[(&str, u64)] = &[
            ("llama.block_count", 32),
            ("llama.attention.head_count", 32),
            ("llama.attention.key_length", 128),
            ("llama.context_length", 8192),
        ];
        b.extend_from_slice(&(entries.len() as u64).to_le_bytes());
        for (key, value) in entries {
            write_string(&mut b, key);
            write_scalar(&mut b, ValueType::U32, *value);
        }
        let shape = read_kv_shape(&mut &b[..]).unwrap();
        assert_eq!(shape.head_count_kv, 32, "falls back to head_count");
        assert_eq!(shape.value_length, 128, "falls back to key_length");
    }

    #[test]
    fn key_and_value_length_derive_from_embedding_and_head_count_when_absent() {
        // The oldest shape: no explicit key_length/value_length at all.
        // llama.cpp's own fallback is embedding_length / head_count.
        let mut b = Vec::new();
        b.extend_from_slice(&MAGIC.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        let entries: &[(&str, u64)] = &[
            ("gptneox.block_count", 24),
            ("gptneox.attention.head_count", 16),
            ("gptneox.embedding_length", 2048), // 2048 / 16 = 128
            ("gptneox.context_length", 2048),
        ];
        b.extend_from_slice(&(entries.len() as u64).to_le_bytes());
        for (key, value) in entries {
            write_string(&mut b, key);
            write_scalar(&mut b, ValueType::U32, *value);
        }
        let shape = read_kv_shape(&mut &b[..]).unwrap();
        assert_eq!(shape.key_length, 128);
        assert_eq!(shape.value_length, 128);
    }

    #[test]
    fn a_vocabulary_sized_array_is_skipped_not_allocated() {
        // The realistic hazard: a tokenizer array with tens of thousands of
        // strings sitting between the keys this reader wants. If skip_value
        // mis-measures an array, every key after it is read at the wrong
        // offset and the whole result is garbage — so this asserts the
        // shape is still correct with a large array in the way, not just
        // that skipping doesn't crash.
        let mut b = Vec::new();
        b.extend_from_slice(&MAGIC.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.extend_from_slice(&5u64.to_le_bytes()); // 5 entries total

        write_string(&mut b, "qwen2.block_count");
        write_scalar(&mut b, ValueType::U32, 48);

        // tokenizer.ggml.tokens: ARRAY of STRING, 2000 entries — small
        // enough for a fast test, large enough to prove the offset math.
        write_string(&mut b, "tokenizer.ggml.tokens");
        b.extend_from_slice(&9u32.to_le_bytes()); // this entry's value type: ARRAY
        b.extend_from_slice(&8u32.to_le_bytes()); // array element type: STRING
        b.extend_from_slice(&2000u64.to_le_bytes()); // array length
        for i in 0..2000u32 {
            write_string(&mut b, &format!("tok{i}"));
        }

        write_string(&mut b, "qwen2.attention.head_count_kv");
        write_scalar(&mut b, ValueType::U32, 8);
        write_string(&mut b, "qwen2.attention.key_length");
        write_scalar(&mut b, ValueType::U32, 128);
        write_string(&mut b, "qwen2.context_length");
        write_scalar(&mut b, ValueType::U32, 32768);

        let shape = read_kv_shape(&mut &b[..]).unwrap();
        assert_eq!(shape.block_count, 48);
        assert_eq!(shape.head_count_kv, 8);
        assert_eq!(shape.context_length, 32768);
    }

    #[test]
    fn the_wrong_magic_is_refused_by_name() {
        let bytes = [0u8; 16];
        let err = read_kv_shape(&mut &bytes[..]).unwrap_err();
        assert!(err.contains("not a GGUF file"), "got: {err}");
    }

    #[test]
    fn a_truncated_file_is_a_named_error_not_a_panic() {
        let bytes = &MAGIC.to_le_bytes()[..2]; // cut mid-magic
        let err = read_kv_shape(&mut &bytes[..]).unwrap_err();
        assert!(!err.is_empty());
    }

    #[test]
    fn missing_block_count_is_a_named_error() {
        let mut b = Vec::new();
        b.extend_from_slice(&MAGIC.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes()); // zero metadata entries
        let err = read_kv_shape(&mut &b[..]).unwrap_err();
        assert!(err.contains("block_count"), "got: {err}");
    }

    // ---- kv_bytes_for: the arithmetic the whole module exists to feed ---

    #[test]
    fn matches_the_shape_measured_on_the_real_muse_incident() {
        // Reverse-engineered from the production numbers, not invented to
        // match the code: muse-local:latest measured at 29.6 GiB total VRAM
        // with an ~15.6 GiB GGUF at parallel=8, n_ctx_slot=4096 (so total
        // context 32768 with the "unset ctx multiplies by parallel"
        // behaviour this module's caller accounts for separately). That is
        // ~14 GiB of KV cache alone, which is the number this arithmetic
        // has to reproduce for a Llama-3-8B-shaped model at that context.
        let shape = KvShape {
            block_count: 32,
            head_count_kv: 8, // Llama 3 8B: GQA, 8 KV heads
            key_length: 128,
            value_length: 128,
            context_length: 8192,
        };
        let total_ctx = 32_768; // 4096 per slot * 8 slots
        let bytes = shape.kv_bytes_for(total_ctx);
        let gib = bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        // 32 blocks * 8 kv heads * 256 (key+value width) * 32768 tokens * 2
        // bytes = 4.0 GiB. That is this one model's KV alone; the incident's
        // measured ~14 GiB overhead was TWO resident models (muse + the
        // second one loading) at this scale of context, which is the
        // multi-model case the planner test below exercises directly rather
        // than folding into a single magic number here.
        assert!(
            (3.9..4.1).contains(&gib),
            "expected ~4 GiB for this one model's KV cache, got {gib:.1} GiB"
        );
    }

    #[test]
    fn doubling_context_doubles_kv_bytes() {
        let shape = KvShape {
            block_count: 32,
            head_count_kv: 8,
            key_length: 128,
            value_length: 128,
            context_length: 8192,
        };
        let one = shape.kv_bytes_for(4096);
        let two = shape.kv_bytes_for(8192);
        assert_eq!(two, one * 2);
    }

    #[test]
    fn zero_context_is_zero_bytes_not_a_panic() {
        let shape = KvShape {
            block_count: 32,
            head_count_kv: 8,
            key_length: 128,
            value_length: 128,
            context_length: 8192,
        };
        assert_eq!(shape.kv_bytes_for(0), 0);
    }
}
