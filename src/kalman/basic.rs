use alga::general::{Inverse, Multiplicative, RingCommutative};
use nalgebra::base::allocator::Allocator;
use nalgebra::base::DefaultAllocator;
use nalgebra::{DimName, MatrixMN, Real, VectorN};

use super::KalmanFilter;
use crate::core::{new_matrix, MatrixNum};

pub struct KalmanGainsController<T: MatrixNum, A: DimName, B: DimName>
where
    DefaultAllocator:
        Allocator<T, A, A> + Allocator<T, A, B> + Allocator<T, B, A> + Allocator<T, B, B>,
    MatrixMN<T, B, B>: Inverse<Multiplicative>,
{
    measurement: MatrixMN<T, B, A>,
    measurement_noise_covariance: MatrixMN<T, B, B>,
    measurement_transpose: MatrixMN<T, A, B>,
}

impl<T: MatrixNum, A: DimName, C: DimName> KalmanGainsController<T, A, C>
where
    DefaultAllocator:
        Allocator<T, A, A> + Allocator<T, A, C> + Allocator<T, C, A> + Allocator<T, C, C>,
    MatrixMN<T, C, C>: Inverse<Multiplicative>,
{
    pub fn new(
        measurement: MatrixMN<T, C, A>,
        measurement_noise_covariance: MatrixMN<T, C, C>,
    ) -> Self {
        KalmanGainsController {
            measurement: measurement.clone(),
            measurement_noise_covariance,
            measurement_transpose: measurement.transpose(),
        }
    }

    pub fn next_gains(&self, error_covariance: MatrixMN<T, A, A>) -> MatrixMN<T, A, C> {
        &error_covariance
            * &self.measurement_transpose
            * nalgebra::inverse(
                &(&self.measurement * error_covariance * &self.measurement_transpose
                    + &self.measurement_noise_covariance),
            )
    }
}

pub struct KalmanStateTransferFunction<T: MatrixNum, A: DimName, C: DimName>
where
    DefaultAllocator: Allocator<T, C, A>,
{
    measurement: MatrixMN<T, C, A>,
}

impl<T: MatrixNum, A: DimName, C: DimName> KalmanStateTransferFunction<T, A, C>
where
    DefaultAllocator: Allocator<T, A, C> + Allocator<T, C, A> + Allocator<T, A> + Allocator<T, C>,
{
    pub fn new(measurement: MatrixMN<T, C, A>) -> Self {
        KalmanStateTransferFunction { measurement }
    }

    #[inline]
    pub fn eval(
        &self,
        kalman_gains: MatrixMN<T, A, C>,
        states: VectorN<T, A>,
        outputs: VectorN<T, C>,
    ) -> VectorN<T, A> {
        &states + kalman_gains * (outputs - &self.measurement * &states)
    }
}

pub struct KalmanErrorCovarianceTransferFunction<T: MatrixNum, A: DimName, C: DimName>
where
    DefaultAllocator: Allocator<T, C, A> + Allocator<T, A, A>,
{
    measurement: MatrixMN<T, C, A>,
    identity: MatrixMN<T, A, A>,
}

impl<T: MatrixNum, A: DimName, C: DimName> KalmanErrorCovarianceTransferFunction<T, A, C>
where
    DefaultAllocator: Allocator<T, A, C> + Allocator<T, C, A> + Allocator<T, A, A>,
{
    pub fn new(measurement: MatrixMN<T, C, A>) -> Self {
        let mut res = unsafe { MatrixMN::new_uninitialized_generic(A::name(), A::name()) };
        res.fill_with_identity();

        KalmanErrorCovarianceTransferFunction {
            measurement,
            identity: res,
        }
    }

    #[inline]
    pub fn eval(
        &self,
        kalman_gains: MatrixMN<T, A, C>,
        process_covariance: MatrixMN<T, A, A>,
    ) -> MatrixMN<T, A, A> {
        (&self.identity - kalman_gains * &self.measurement) * process_covariance
    }
}

pub struct BasicKalmanFilter<T: Real + RingCommutative, A: DimName, B: DimName, C: DimName>
where
    DefaultAllocator: Allocator<T, A, A>
        + Allocator<T, A, B>
        + Allocator<T, A, C>
        + Allocator<T, C, A>
        + Allocator<T, C, C>
        + Allocator<T, B, A>
        + Allocator<T, B, B>
        + Allocator<T, A>
        + Allocator<T, B>
        + Allocator<T, C>,
    MatrixMN<T, C, C>: Inverse<Multiplicative>,
{
    kalman_gains: KalmanGainsController<T, A, C>,
    state_transfer: KalmanStateTransferFunction<T, A, C>,
    error_covariance_transfer: KalmanErrorCovarianceTransferFunction<T, A, C>,
    error_covariance: MatrixMN<T, A, A>,
    process_noise: MatrixMN<T, A, A>,
    system: MatrixMN<T, A, A>,
    input: MatrixMN<T, A, B>,
}

impl<T: MatrixNum, A: DimName, B: DimName, C: DimName> BasicKalmanFilter<T, A, B, C>
where
    DefaultAllocator: Allocator<T, A, A>
        + Allocator<T, A, B>
        + Allocator<T, B, A>
        + Allocator<T, B, B>
        + Allocator<T, A>
        + Allocator<T, B>
        + Allocator<T, A, C>
        + Allocator<T, C, A>
        + Allocator<T, C, C>
        + Allocator<T, C>,
    MatrixMN<T, C, C>: Inverse<Multiplicative>,
{
    pub fn new(
        system: MatrixMN<T, A, A>,
        input: MatrixMN<T, A, B>,
        process_noise: MatrixMN<T, A, A>,
        measurement: MatrixMN<T, C, A>,
        measurement_noise_covariance: MatrixMN<T, C, C>,
    ) -> Self {
        BasicKalmanFilter {
            kalman_gains: KalmanGainsController::new(
                measurement.clone(),
                measurement_noise_covariance,
            ),
            state_transfer: KalmanStateTransferFunction::new(measurement.clone()),
            error_covariance_transfer: KalmanErrorCovarianceTransferFunction::new(measurement),
            error_covariance: new_matrix(),
            process_noise,
            system,
            input,
        }
    }
}

impl<T: MatrixNum, A: DimName, B: DimName, C: DimName> KalmanFilter<T, A, B, C>
    for BasicKalmanFilter<T, A, B, C>
where
    DefaultAllocator: Allocator<T, A, A>
        + Allocator<T, A, B>
        + Allocator<T, B, A>
        + Allocator<T, B, B>
        + Allocator<T, A>
        + Allocator<T, B>
        + Allocator<T, A, C>
        + Allocator<T, C, A>
        + Allocator<T, C, C>
        + Allocator<T, C>,
    MatrixMN<T, C, C>: Inverse<Multiplicative>,
{
    fn update(
        &mut self,
        state: VectorN<T, A>,
        inputs: VectorN<T, B>,
        outputs: VectorN<T, C>,
    ) -> VectorN<T, A> {
        let prediction_state = &self.system * &state + &self.input * inputs;

        let prediction_error_covariance =
            &self.system * &self.error_covariance * self.system.transpose() + &self.process_noise;

        let kalman_gains = self
            .kalman_gains
            .next_gains(prediction_error_covariance.clone());
        let new_state_estimate =
            self.state_transfer
                .eval(kalman_gains.clone(), prediction_state, outputs);
        self.error_covariance = self
            .error_covariance_transfer
            .eval(kalman_gains, prediction_error_covariance);
        new_state_estimate
    }
}
