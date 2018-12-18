use alga::general::{Inverse, Multiplicative, RingCommutative};
use nalgebra::base::allocator::Allocator;
use nalgebra::base::DefaultAllocator;
use nalgebra::{DimName, MatrixMN, Real, Scalar, VectorN};

pub mod kalman;

pub trait StateSpaceNotation<T: Real + Scalar + RingCommutative, A: DimName, B: DimName, C: DimName>
where
    DefaultAllocator: Allocator<T, A> + Allocator<T, B> + Allocator<T, C>,
{
    fn next_output(
        &self,
        states: VectorN<T, A>,
        inputs: VectorN<T, B>,
    ) -> VectorN<T, C>;
    fn next_state(
        &self,
        states: VectorN<T, A>,
        inputs: VectorN<T, B>,
    ) -> VectorN<T, A>;
}

/// A general fixed controller in state space notation.
pub struct ContinuousTimeInvariant<
    T: Real + Scalar + RingCommutative,
    A: DimName,
    B: DimName,
    C: DimName,
> where
    DefaultAllocator:
        Allocator<T, A, A> + Allocator<T, A, B> + Allocator<T, C, A> + Allocator<T, C, B>,
{
    system: MatrixMN<T, A, A>,
    input: MatrixMN<T, A, B>,
    output: MatrixMN<T, C, A>,
    feedthrough: MatrixMN<T, C, B>,
}

impl<T: Real + Scalar + RingCommutative, A: DimName, B: DimName, C: DimName>
    StateSpaceNotation<T, A, B, C> for ContinuousTimeInvariant<T, A, B, C>
where
    DefaultAllocator: Allocator<T, A, A>
        + Allocator<T, A, B>
        + Allocator<T, C, A>
        + Allocator<T, C, B>
        + Allocator<T, A>
        + Allocator<T, B>
        + Allocator<T, C>,
{
    fn next_output(
        &self,
        states: VectorN<T, A>,
        inputs: VectorN<T, B>,
    ) -> VectorN<T, C> {
        &self.output * states + &self.feedthrough * inputs
    }

    fn next_state(
        &self,
        states: VectorN<T, A>,
        inputs: VectorN<T, B>,
    ) -> VectorN<T, A> {
        &self.system * states + &self.input * inputs
    }
}

impl<T: Real + Scalar + RingCommutative, A: DimName, B: DimName, C: DimName>
    ContinuousTimeInvariant<T, A, B, C>
where
    DefaultAllocator:
        Allocator<T, A, A> + Allocator<T, A, B> + Allocator<T, C, A> + Allocator<T, C, B>,
{
    pub fn new(
        system: MatrixMN<T, A, A>,
        input: MatrixMN<T, A, B>,
        output: MatrixMN<T, C, A>,
        feedthrough: MatrixMN<T, C, B>,
    ) -> Self {
        ContinuousTimeInvariant {
            system,
            input,
            output,
            feedthrough,
        }
    }
}

//pub struct KalmanHelper<T: Real + RingCommutative, A: DimName, B: DimName, C: DimName>
//    where
//        DefaultAllocator: Allocator<T, A, A>
//        + Allocator<T, A, B>
//        + Allocator<T, A, C>
//        + Allocator<T, C, A>
//        + Allocator<T, C, C>
//        + Allocator<T, B, A>
//        + Allocator<T, B, B>
//        + Allocator<T, A>
//        + Allocator<T, B>
//        + Allocator<T, C>,
//        MatrixMN<T, C, C>: Inverse<Multiplicative>,
//{
//    filter: kalman::KalmanFilter<T, A, B, C>,
//}
//
//impl<T: Real + RingCommutative, A: DimName, B: DimName, C: DimName> KalmanHelper<T, A, B, C> {
//
//}

pub fn covariance<T: Real>(a: Vec<T>, b: Vec<T>) -> T {
    let len = a.len().min(b.len());
    let mut mean_a = T::zero();
    let mut mean_b = T::zero();

    for index in 0..len {
        mean_a += a[index];
        mean_b += b[index];
    }

    mean_a /= len;
    mean_b /= len;

    let mut covariance = T::zero();

    for index in 0..len {
        covariance += (a[index] - mean_a) * (b[index] - mean_b);
    }

    match T::from_usize(len) {
        Some(v) => covariance / v,
        None => T::zero()
    }
}