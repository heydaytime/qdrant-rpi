//! Vector scaling functions for RPI
//!
//! The core geometric operation: scale vectors by shell number k.
//! When query and stored vectors are both scaled by k, Euclidean distance
//! scales by exactly k: |k·q − k·v| = k·|q − v|

use segment::data_types::vectors::DenseVector;

/// Scale a vector by a shell factor k.
///
/// This is the fundamental RPI operation: store vectors at k * embedding
/// to encode quality geometrically.
///
/// # Arguments
/// * `vector` - The original embedding vector
/// * `k` - The shell number (1 = highest quality, higher = lower quality)
///
/// # Returns
/// A new vector with each component multiplied by k
#[inline]
pub fn scale_vector(vector: &[f32], k: u8) -> DenseVector {
    let k_f32 = f32::from(k);
    vector.iter().map(|&x| x * k_f32).collect()
}

/// Scale a vector by a floating point factor.
/// Used for unscaling (dividing by k) when promoting or extracting original vectors.
#[inline]
pub fn scale_vector_f32(vector: &[f32], factor: f32) -> DenseVector {
    vector.iter().map(|&x| x * factor).collect()
}

/// Unscale a vector to recover the original embedding.
///
/// Given a vector stored at shell k (stored as k * original),
/// recover the original by dividing by k.
#[inline]
pub fn unscale_vector(scaled_vector: &[f32], k: u8) -> DenseVector {
    if k == 0 {
        // Avoid division by zero, return as-is
        scaled_vector.to_vec()
    } else {
        let k_f32 = f32::from(k);
        scaled_vector.iter().map(|&x| x / k_f32).collect()
    }
}

/// Calculate the magnitude (L2 norm) of a vector.
/// This can be used to infer which shell a vector is stored at
/// if the original embedding was normalized.
#[inline]
pub fn vector_magnitude(vector: &[f32]) -> f32 {
    vector.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

/// Infer the shell number from a scaled vector, given the expected
/// magnitude of the original normalized embedding.
///
/// For normalized embeddings (magnitude ≈ 1), the shell is simply
/// the magnitude of the scaled vector.
///
/// Returns the nearest integer shell number.
pub fn infer_shell_from_magnitude(vector: &[f32]) -> u8 {
    let mag = vector_magnitude(vector);
    // Round to nearest integer, minimum 1
    (mag.round() as u8).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_vector() {
        let v = vec![1.0, 2.0, 3.0];

        let scaled_1 = scale_vector(&v, 1);
        assert_eq!(scaled_1, vec![1.0, 2.0, 3.0]);

        let scaled_2 = scale_vector(&v, 2);
        assert_eq!(scaled_2, vec![2.0, 4.0, 6.0]);

        let scaled_5 = scale_vector(&v, 5);
        assert_eq!(scaled_5, vec![5.0, 10.0, 15.0]);
    }

    #[test]
    fn test_unscale_vector() {
        let original = vec![1.0, 2.0, 3.0];
        let scaled = scale_vector(&original, 3);
        let recovered = unscale_vector(&scaled, 3);

        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_euclidean_distance_scaling() {
        // This test verifies the core RPI property:
        // |k·q − k·v| = k·|q − v|

        let q = vec![1.0, 0.0, 0.0];
        let v = vec![0.0, 1.0, 0.0];

        // Original distance
        let dist_1: f32 = q
            .iter()
            .zip(v.iter())
            .map(|(a, b)| {
                let d: f32 = a - b;
                d * d
            })
            .sum::<f32>()
            .sqrt();

        // Distance at shell k=3
        let q_scaled = scale_vector(&q, 3);
        let v_scaled = scale_vector(&v, 3);
        let dist_3: f32 = q_scaled
            .iter()
            .zip(v_scaled.iter())
            .map(|(a, b)| {
                let d: f32 = a - b;
                d * d
            })
            .sum::<f32>()
            .sqrt();

        // dist_3 should be exactly 3 * dist_1
        assert!((dist_3 - 3.0 * dist_1).abs() < 1e-6);
    }

    #[test]
    fn test_magnitude() {
        // Unit vector
        let v = vec![1.0, 0.0, 0.0];
        assert!((vector_magnitude(&v) - 1.0).abs() < 1e-6);

        // Scaled unit vector
        let v_scaled = scale_vector(&v, 5);
        assert!((vector_magnitude(&v_scaled) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_infer_shell() {
        let unit = vec![1.0, 0.0, 0.0];

        assert_eq!(infer_shell_from_magnitude(&scale_vector(&unit, 1)), 1);
        assert_eq!(infer_shell_from_magnitude(&scale_vector(&unit, 3)), 3);
        assert_eq!(infer_shell_from_magnitude(&scale_vector(&unit, 5)), 5);
    }
}
