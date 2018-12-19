use alga::general::RingCommutative;
use nalgebra::base::allocator::Allocator;
use nalgebra::base::DefaultAllocator;
use nalgebra::{DimName, MatrixMN, Real, Scalar, VectorN};

/// Basic number type used when dealing with matrices
pub trait MatrixNum: Real + Scalar + RingCommutative {}

impl<T: Real + Scalar + RingCommutative> MatrixNum for T {}

pub trait SingleDerive<T: MatrixNum, N: DimName, R: DimName, C: DimName>
where
    DefaultAllocator: Allocator<T, N> + Allocator<T, R, C>,
{
    fn derive(vec: VectorN<T, N>) -> MatrixMN<T, R, C>;
}
