mod ndarray_conv_binds;

use crate::{Error, Result};
use alloc::string::ToString;
use ndarray::{Array1, ArrayView1};
use ndarray_conv::{ConvExt, ConvFFTExt, PaddingMode};

/// Convolution mode determines behavior near edges and output size
pub enum ConvolveMode {
    /// Full convolution, output size is `in1.len() + in2.len() - 1`
    Full,
    /// Valid convolution, output size is `max(in1.len(), in2.len()) - min(in1.len(), in2.len()) + 1`
    Valid,
    /// Same convolution, output size is `in1.len()`
    Same,
}

/// Best effort parallel behaviour with numpy's convolve method. We take `v` as the convolution
/// kernel.
///
/// Returns the discrete, linear convolution of two one-dimensional sequences.
///
/// # Parameters
/// * `a` : (N,) [[array_like]]([ndarray::Array1])  
///   Signal to be (linearly) convolved.
/// * `v` : (M,) [[array_like]]([ndarray::Array1])  
///   Second one-dimensional input array.
/// * `mode` : [ConvolveMode]  
///   [ConvolveMode::Full]:  
///   By default, mode is 'full'.  This returns the convolution at each point of overlap, with an
///   output shape of (N+M-1,). At the end-points of the convolution, the signals do not overlap
///   completely, and boundary effects may be seen.
///
///   [ConvolveMode::Same]:  
///   Mode 'same' returns output of length ``max(M, N)``.  Boundary effects are still visible.
///
///   [ConvolveMode::Valid]:  
///   Mode 'valid' returns output of length ``max(M, N) - min(M, N) + 1``.  The convolution
///   product is only given for points where the signals overlap completely.  Values outside the
///   signal boundary have no effect.
///
/// # Panics
/// We assume that `v` is shorter than `a`.
///
/// # Examples
/// With [ConvolveMode::Full]:
/// ```
/// use ndarray::array;
/// use sci_rs_core::num_rs::{ConvolveMode, convolve};
///
/// let a = array![1., 2., 3.];
/// let v = array![0., 1., 0.5];
///
/// let expected = array![0., 1., 2.5, 4., 1.5];
/// let result = convolve((&a).into(), (&v).into(), ConvolveMode::Full).unwrap();
/// assert_eq!(result, expected);
/// ```
/// With [ConvolveMode::Same]:
/// ```
/// use ndarray::array;
/// use sci_rs_core::num_rs::{ConvolveMode, convolve};
///
/// let a = array![1., 2., 3.];
/// let v = array![0., 1., 0.5];
///
/// let expected = array![1., 2.5, 4.];
/// let result = convolve((&a).into(), (&v).into(), ConvolveMode::Same).unwrap();
/// assert_eq!(result, expected);
/// ```
/// With [ConvolveMode::Same]:
/// ```
/// use ndarray::array;
/// use sci_rs_core::num_rs::{ConvolveMode, convolve};
///
/// let a = array![1., 2., 3.];
/// let v = array![0., 1., 0.5];
///
/// let expected = array![2.5];
/// let result = convolve((&a).into(), (&v).into(), ConvolveMode::Valid).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn convolve<T>(a: ArrayView1<T>, v: ArrayView1<T>, mode: ConvolveMode) -> Result<Array1<T>>
where
    T: num_traits::NumAssign + core::marker::Copy,
{
    // Convolve
    let result = a.conv(&v, mode.into(), PaddingMode::Zeros);
    #[cfg(feature = "alloc")]
    {
        result.map_err(|e| Error::Conv {
            reason: e.to_string(),
        })
    }
    #[cfg(not(feature = "alloc"))]
    {
        result.map_err({ Error::Conv })
    }
}

/// Best effort parallel behaviour with numpy's convolve method. We take `v` as the convolution
/// kernel, with scratch space for kernel.
///
/// Returns the discrete, linear convolution of two one-dimensional sequences.
///
/// # Parameters
/// * `a` : (N,) [[array_like]]([ndarray::Array1])  
///   Signal to be (linearly) convolved.
/// * `v` : (M,) [[array_like]]([ndarray::Array1])  
///   Second one-dimensional input array.
/// * `mode` : [ConvolveMode]  
///   [ConvolveMode::Full]:  
///   By default, mode is 'full'.  This returns the convolution at each point of overlap, with an
///   output shape of (N+M-1,). At the end-points of the convolution, the signals do not overlap
///   completely, and boundary effects may be seen.
///
///   [ConvolveMode::Same]:  
///   Mode 'same' returns output of length ``max(M, N)``.  Boundary effects are still visible.
///
///   [ConvolveMode::Valid]:  
///   Mode 'valid' returns output of length ``max(M, N) - min(M, N) + 1``.  The convolution
///   product is only given for points where the signals overlap completely.  Values outside the
///   signal boundary have no effect.
///
/// # Panics
/// We assume that `v` is shorter than `a`.
///
/// # Examples
/// With [ConvolveMode::Full]:
/// ```
/// use ndarray::array;
/// use sci_rs_core::num_rs::{ConvolveMode, convolve_scratchf64, prelude::get_fft_processor};
///
/// let a = array![1., 2., 3.];
/// let v = array![0., 1., 0.5];
/// let mut proc = get_fft_processor();
///
/// let expected = array![0., 1., 2.5, 4., 1.5];
/// let result = convolve_scratchf64(a.view(), v.view(), ConvolveMode::Full, &mut proc).unwrap();
///
/// use approx::assert_relative_eq;
/// use ndarray::Zip;
/// Zip::from(&result)
///     .and(&expected)
///     .for_each(|&r, &e| assert_relative_eq!(r, e, max_relative = 1e-7, epsilon = 1e-12));
/// ```
/// With [ConvolveMode::Same]:
/// ```
/// use ndarray::array;
/// use sci_rs_core::num_rs::{ConvolveMode, convolve_scratchf64, prelude::get_fft_processor};
///
/// let a = array![1., 2., 3.];
/// let v = array![0., 1., 0.5];
/// let mut proc = get_fft_processor();
///
/// let expected = array![1., 2.5, 4.];
/// let result = convolve_scratchf64(a.view(), v.view(), ConvolveMode::Same, &mut proc).unwrap();
///
/// use approx::assert_relative_eq;
/// use ndarray::Zip;
/// Zip::from(&result)
///     .and(&expected)
///     .for_each(|&r, &e| assert_relative_eq!(r, e, max_relative = 1e-7, epsilon = 1e-12));
/// ```
/// With [ConvolveMode::Same]:
/// ```
/// use ndarray::array;
/// use sci_rs_core::num_rs::{ConvolveMode, convolve_scratchf64, prelude::get_fft_processor};
///
/// let a = array![1., 2., 3.];
/// let v = array![0., 1., 0.5];
/// let mut proc = get_fft_processor();
///
/// let expected = array![2.5];
/// let result = convolve_scratchf64(a.view(), v.view(), ConvolveMode::Valid, &mut proc).unwrap();
///
/// use approx::assert_relative_eq;
/// use ndarray::Zip;
/// Zip::from(&result)
///     .and(&expected)
///     .for_each(|&r, &e| assert_relative_eq!(r, e, max_relative = 1e-7, epsilon = 1e-12));
/// ```
pub fn convolve_scratchf64(
    a: ArrayView1<f64>,
    v: ArrayView1<f64>,
    mode: ConvolveMode,
    proc: &mut impl ndarray_conv::FftProcessor<f64, f64>,
) -> Result<Array1<f64>> {
    // Convolve
    let result = a.conv_fft_with_processor(&v, mode.into(), PaddingMode::Zeros, proc);
    #[cfg(feature = "alloc")]
    {
        result.map_err(|e| Error::Conv {
            reason: e.to_string(),
        })
    }
    #[cfg(not(feature = "alloc"))]
    {
        result.map_err({ Error::Conv })
    }
}

#[cfg(test)]
mod linear_convolve {
    use super::*;
    use alloc::vec;
    use ndarray::array;

    #[test]
    fn full() {
        let a = array![1., 2., 3.];
        let v = array![0., 1., 0.5];

        let expected = array![0., 1., 2.5, 4., 1.5];
        let result = convolve((&a).into(), (&v).into(), ConvolveMode::Full).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn same() {
        let a = array![1., 2., 3.];
        let v = array![0., 1., 0.5];

        let expected = array![1., 2.5, 4.];
        let result = convolve((&a).into(), (&v).into(), ConvolveMode::Same).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn valid() {
        let a = array![1., 2., 3.];
        let v = array![0., 1., 0.5];

        let expected = array![2.5];
        let result = convolve((&a).into(), (&v).into(), ConvolveMode::Valid).unwrap();
        assert_eq!(result, expected);
    }
}

#[cfg(test)]
mod fft64_convolve {
    use super::*;
    use alloc::vec;
    use approx::assert_relative_eq;
    use ndarray::{array, Zip};
    use ndarray_conv::get_fft_processor;

    #[test]
    fn full() {
        let a = array![1., 2., 3.];
        let v = array![0., 1., 0.5];
        let mut proc = get_fft_processor::<_, _>();

        let expected = array![0., 1., 2.5, 4., 1.5];
        let result =
            convolve_scratchf64((&a).into(), (&v).into(), ConvolveMode::Full, &mut proc).unwrap();
        Zip::from(&expected)
            .and(&result)
            .for_each(|&e, &r| assert_relative_eq!(r, e));
    }

    #[test]
    fn same() {
        let a = array![1., 2., 3.];
        let v = array![0., 1., 0.5];
        let mut proc = get_fft_processor::<_, _>();

        let expected = array![1., 2.5, 4.];
        let result =
            convolve_scratchf64((&a).into(), (&v).into(), ConvolveMode::Same, &mut proc).unwrap();
        Zip::from(&expected)
            .and(&result)
            .for_each(|&e, &r| assert_relative_eq!(r, e));
    }

    #[test]
    fn valid() {
        let a = array![1., 2., 3.];
        let v = array![0., 1., 0.5];
        let mut proc = get_fft_processor::<_, _>();

        let expected = array![2.5];
        let result =
            convolve_scratchf64((&a).into(), (&v).into(), ConvolveMode::Valid, &mut proc).unwrap();
        Zip::from(&expected)
            .and(&result)
            .for_each(|&e, &r| assert_relative_eq!(r, e));
    }
}
