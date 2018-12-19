use nalgebra::base::allocator::Allocator;
use nalgebra::base::DefaultAllocator;
use nalgebra::{DimName, VectorN};

use crate::core::MatrixNum;

pub mod basic;

trait KalmanFilter<T: MatrixNum, A: DimName, B: DimName, C: DimName>
where
    DefaultAllocator: Allocator<T, A> + Allocator<T, B> + Allocator<T, C>,
{
    fn update(
        &mut self,
        state: VectorN<T, A>,
        inputs: VectorN<T, B>,
        outputs: VectorN<T, C>,
    ) -> VectorN<T, A>;
}
