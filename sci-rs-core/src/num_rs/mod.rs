#[cfg(feature = "alloc")]
mod convolve;
#[cfg(feature = "alloc")]
pub use convolve::*;

pub mod prelude {
    #[cfg(feature = "alloc")]
    pub use ndarray_conv::{get_fft_processor, FftProcessor};
}
