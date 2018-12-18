use alga::general::RingCommutative;
use nalgebra::base::allocator::Allocator;
use nalgebra::base::DefaultAllocator;
use nalgebra::{DimName, MatrixMN, Real, Scalar, VectorN};

mod core;
pub mod kalman;

pub use crate::core::*;

pub trait StateSpaceNotation<T: Real + Scalar + RingCommutative, A: DimName, B: DimName, C: DimName>
where
    DefaultAllocator: Allocator<T, A> + Allocator<T, B> + Allocator<T, C>,
{
    fn next_output(&self, states: VectorN<T, A>, inputs: VectorN<T, B>) -> VectorN<T, C>;
    fn next_state(&self, states: VectorN<T, A>, inputs: VectorN<T, B>) -> VectorN<T, A>;
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
    fn next_output(&self, states: VectorN<T, A>, inputs: VectorN<T, B>) -> VectorN<T, C> {
        &self.output * states + &self.feedthrough * inputs
    }

    fn next_state(&self, states: VectorN<T, A>, inputs: VectorN<T, B>) -> VectorN<T, A> {
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
