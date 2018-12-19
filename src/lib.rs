use alga::general::RingCommutative;
use nalgebra::base::allocator::Allocator;
use nalgebra::base::DefaultAllocator;
use nalgebra::{DimName, MatrixMN, Real, Scalar, VectorN};

pub mod core;
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

//// Work in progress
//pub struct ExtendedKalmanFilter<T: Real + Scalar + RingCommutative, A: DimName, F: FnMut(T) -> T> {
//    state_functions: Vec<F>,
//    measurement_functions: Vec<F>,
//    process_noise_covariance: MatrixMN<T, A, A>,
//    input_noise_variance2: MatrixMN<T, A, A>,
//}
//
//use crate::core::jacobian_square;
//impl<T: Real + Scalar + RingCommutative, A: DimName, B: DimName, C: DimName, F: FnMut(T) -> T>
//    ExtendedKalmanFilter<T, A, F>
//where
//    DefaultAllocator: Allocator<T, A, A>
//        + Allocator<T, A, B>
//        + Allocator<T, C, A>
//        + Allocator<T, C, B>
//        + Allocator<T, A>
//        + Allocator<T, B>
//        + Allocator<T, C>,
//{
//    pub fn new(state_functions: Vec<F>) {}
//
//    //    pub fn predict_state(
//    //        &self,
//    //        state_estimate: VectorN<T, A>,
//    //        inputs: VectorN<T, B>,
//    //    ) -> VectorN<T, A> {
//    //        &self.system * state_estimate + &self.input * inputs
//    //    }
//
//    pub fn predict_error_covariance(
//        &mut self,
//        error_covariance: MatrixMN<T, A, A>,
//        state: VectorN<T, A>,
//        inputs: VectorN<T, B>,
//    ) -> MatrixMN<T, A, A> {
//        //        &self.system * error_covariance * self.system.transpose() + &self.process_noise
//        let system = self.system(state);
//        let input = self.input(inputs);
//
//        // P(k)-
//        system * error_covariance * system.transpose()
//            + self.input_noise_variance2 * input * input.transpose()
//            + self.process_noise_covariance
//    }
//
//    /// A(k)
//    fn system(&mut self, system_state: &mut VectorN<T, A>) -> MatrixMN<T, A, A> {
//        jacobian_square(system_state, &mut self.state_functions)
//    }
//
//    /// B(k)
//    fn input(&mut self, inputs: &mut VectorN<T, B>) -> MatrixMN<T, A, B> {
//        jacobian_square(inputs, &mut self.state_functions)
//    }
//
//    /// H(k)
//    fn measurement(&mut self, measurement: &mut VectorN<T, A>) -> MatrixMN<T, C, A> {
//        jacobian_square(measurement, &mut self.measurement_functions)
//    }
//}
