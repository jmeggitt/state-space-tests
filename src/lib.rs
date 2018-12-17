use alga::general::{Inverse, Multiplicative, RingCommutative};
use nalgebra::base::allocator::Allocator;
use nalgebra::base::DefaultAllocator;
use nalgebra::{DimName, MatrixMN, Real, Scalar, U1};

pub trait StateSpaceNotation<T: Real + Scalar + RingCommutative, A: DimName, B: DimName, C: DimName>
where
    DefaultAllocator: Allocator<T, A, U1> + Allocator<T, B, U1> + Allocator<T, C, U1>,
{
    fn next_output(
        &self,
        states: MatrixMN<T, A, U1>,
        inputs: MatrixMN<T, B, U1>,
    ) -> MatrixMN<T, C, U1>;
    fn next_state(
        &self,
        states: MatrixMN<T, A, U1>,
        inputs: MatrixMN<T, B, U1>,
    ) -> MatrixMN<T, A, U1>;
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
        + Allocator<T, A, U1>
        + Allocator<T, B, U1>
        + Allocator<T, C, U1>,
{
    fn next_output(
        &self,
        states: MatrixMN<T, A, U1>,
        inputs: MatrixMN<T, B, U1>,
    ) -> MatrixMN<T, C, U1> {
        &self.output * states + &self.feedthrough * inputs
    }

    fn next_state(
        &self,
        states: MatrixMN<T, A, U1>,
        inputs: MatrixMN<T, B, U1>,
    ) -> MatrixMN<T, A, U1> {
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

pub struct KalmanGainsController<T: Real + Scalar + RingCommutative, A: DimName, B: DimName>
where
    DefaultAllocator:
        Allocator<T, A, A> + Allocator<T, A, B> + Allocator<T, B, A> + Allocator<T, B, B>,
    MatrixMN<T, B, B>: Inverse<Multiplicative>,
{
    measurement: MatrixMN<T, B, A>,
    measurement_noise_covariance: MatrixMN<T, B, B>,
    /// Pre-compute to save time
    measurement_transpose: MatrixMN<T, A, B>,
}

impl<T: Real + Scalar + RingCommutative, A: DimName, C: DimName> KalmanGainsController<T, A, C>
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

pub struct KalmanStateTransferFunction<T: Real + Scalar + RingCommutative, A: DimName, C: DimName>
where
    DefaultAllocator: Allocator<T, C, A>,
{
    measurement: MatrixMN<T, C, A>,
}

impl<T: Real + Scalar + RingCommutative, A: DimName, C: DimName>
    KalmanStateTransferFunction<T, A, C>
where
    DefaultAllocator:
        Allocator<T, A, C> + Allocator<T, C, A> + Allocator<T, A, U1> + Allocator<T, C, U1>,
{
    pub fn new(measurement: MatrixMN<T, C, A>) -> Self {
        KalmanStateTransferFunction { measurement }
    }

    #[inline]
    pub fn eval(
        &self,
        kalman_gains: MatrixMN<T, A, C>,
        states: MatrixMN<T, A, U1>,
        outputs: MatrixMN<T, C, U1>,
    ) -> MatrixMN<T, A, U1> {
        &states + kalman_gains * (outputs - &self.measurement * &states)
    }
}

pub struct KalmanErrorCovarianceTransferFunction<
    T: Real + Scalar + RingCommutative,
    A: DimName,
    C: DimName,
> where
    DefaultAllocator: Allocator<T, C, A> + Allocator<T, A, A>,
{
    measurement: MatrixMN<T, C, A>,
    identity: MatrixMN<T, A, A>,
}

impl<T: Real + Scalar + RingCommutative, A: DimName, C: DimName>
    KalmanErrorCovarianceTransferFunction<T, A, C>
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

pub struct KalmanFilter<T: Real + RingCommutative, A: DimName, B: DimName, C: DimName>
where
    DefaultAllocator: Allocator<T, A, A>
        + Allocator<T, A, B>
        + Allocator<T, A, C>
        + Allocator<T, C, A>
        + Allocator<T, C, C>
        + Allocator<T, B, A>
        + Allocator<T, B, B>
        + Allocator<T, A, U1>
        + Allocator<T, B, U1>
        + Allocator<T, C, U1>,
    MatrixMN<T, C, C>: Inverse<Multiplicative>,
{
    kalman_gains: KalmanGainsController<T, A, C>,
    state_transfer: KalmanStateTransferFunction<T, A, C>,
    error_covariance_transfer: KalmanErrorCovarianceTransferFunction<T, A, C>,
    process_noise: MatrixMN<T, A, A>,
    system: MatrixMN<T, A, A>,
    input: MatrixMN<T, A, B>,
}

impl<T: Real + Scalar + RingCommutative, A: DimName, B: DimName, C: DimName>
    KalmanFilter<T, A, B, C>
where
    DefaultAllocator: Allocator<T, A, A>
        + Allocator<T, A, B>
        + Allocator<T, B, A>
        + Allocator<T, B, B>
        + Allocator<T, A, U1>
        + Allocator<T, B, U1>
        + Allocator<T, A, C>
        + Allocator<T, C, A>
        + Allocator<T, C, C>
        + Allocator<T, C, U1>,
    MatrixMN<T, C, C>: Inverse<Multiplicative>,
{
    pub fn new(
        system: MatrixMN<T, A, A>,
        input: MatrixMN<T, A, B>,
        process_noise: MatrixMN<T, A, A>,
        measurement: MatrixMN<T, C, A>,
        measurement_noise_covariance: MatrixMN<T, C, C>,
    ) -> Self {
        KalmanFilter {
            kalman_gains: KalmanGainsController::new(
                measurement.clone(),
                measurement_noise_covariance,
            ),
            state_transfer: KalmanStateTransferFunction::new(measurement.clone()),
            error_covariance_transfer: KalmanErrorCovarianceTransferFunction::new(measurement),
            process_noise,
            system,
            input,
        }
    }

    pub fn predict_state(
        &self,
        state_estimate: MatrixMN<T, A, U1>,
        inputs: MatrixMN<T, B, U1>,
    ) -> MatrixMN<T, A, U1> {
        &self.system * state_estimate + &self.input * inputs
    }

    pub fn predict_error_covariance(
        &self,
        error_covariance: MatrixMN<T, A, A>,
    ) -> MatrixMN<T, A, A> {
        &self.system * error_covariance * self.system.transpose() + &self.process_noise
    }

    pub fn update(
        &self,
        error_covariance: MatrixMN<T, A, A>,
        state_estimates: MatrixMN<T, A, U1>,
        outputs: MatrixMN<T, C, U1>,
    ) -> (MatrixMN<T, A, U1>, MatrixMN<T, A, A>) {
        let kalman_gains = self.kalman_gains.next_gains(error_covariance.clone());
        let new_state_estimate =
            self.state_transfer
                .eval(kalman_gains.clone(), state_estimates, outputs);
        let new_error_covariance = self
            .error_covariance_transfer
            .eval(kalman_gains, error_covariance);
        (new_state_estimate, new_error_covariance)
    }
}
