//! Radial Priority Indexing (RPI) - A Self-Organizing, Quality-Convergent Semantic Cache
//!
//! RPI encodes quality geometrically by storing vectors at different "shells" based on their
//! trustworthiness. Vectors close to the origin (k=1) are trusted, vectors far from the origin
//! are garbage waiting to be evicted.
//!
//! # Shell Structure
//! - k=1: Fully validated, trusted data (HNSW indexed)
//! - k=2..N-1: Demoted data (brute force search)
//! - k=N: Eviction threshold - data is permanently deleted
//!
//! # Key Properties
//! - Direction (angle) encodes semantic meaning
//! - Magnitude (radius) encodes quality/trustworthiness
//! - Uses Euclidean distance (NOT cosine) to preserve shell structure
//! - ε_k = ε_1 × k for proper angular coverage at each shell

pub mod config;
pub mod demotion;
pub mod operations;
pub mod scaling;
pub mod search;
pub mod tracking;

pub use config::*;
pub use demotion::*;
pub use operations::*;
pub use scaling::*;
pub use search::*;
pub use tracking::*;

/// Reserved payload field names for RPI metadata
pub mod payload_fields {
    /// The shell this point originally entered at (always 1 for new points)
    pub const ORIGINAL_SHELL: &str = "_rpi_original_shell";
    /// The current shell where this point resides
    pub const CURRENT_SHELL: &str = "_rpi_current_shell";
    /// Number of times this point has been demoted
    pub const DEMOTION_COUNT: &str = "_rpi_demotions";
    /// LRU hit counter - incremented on each access
    pub const HIT_COUNT: &str = "_rpi_hits";
    /// Last access timestamp (unix epoch seconds)
    pub const LAST_ACCESS: &str = "_rpi_last_access";
}

/// Prefix for shell vector names
pub const SHELL_VECTOR_PREFIX: &str = "rpi_shell_";

/// Generate the vector name for a given shell number
pub fn shell_vector_name(shell: u8) -> String {
    format!("{}{}", SHELL_VECTOR_PREFIX, shell)
}

/// Parse shell number from a vector name, returns None if not an RPI shell vector
pub fn parse_shell_number(vector_name: &str) -> Option<u8> {
    vector_name
        .strip_prefix(SHELL_VECTOR_PREFIX)
        .and_then(|s| s.parse().ok())
}

/// Check if a vector name is an RPI shell vector
pub fn is_shell_vector(vector_name: &str) -> bool {
    vector_name.starts_with(SHELL_VECTOR_PREFIX)
}
