use alga::general::RingCommutative;
use nalgebra::base::allocator::Allocator;
use nalgebra::base::DefaultAllocator;
use nalgebra::{DimName, MatrixMN, Real, Scalar, VectorN};

#[inline(always)]
pub fn new_matrix<T: Real + Scalar + RingCommutative, R: DimName, C: DimName>() -> MatrixMN<T, R, C>
where
    DefaultAllocator: Allocator<T, R, C>,
{
    unsafe { MatrixMN::new_uninitialized_generic(R::name(), C::name()) }
}

#[inline(always)]
pub fn apply_new<T: Real + Scalar + RingCommutative, R: DimName, C: DimName, F: FnMut(T) -> T>(
    matrix: &MatrixMN<T, R, C>,
    function: F,
) -> MatrixMN<T, R, C>
where
    DefaultAllocator: Allocator<T, R, C>,
{
    let mut new_matrix = matrix.clone();
    new_matrix.apply(function);
    new_matrix
}

#[inline]
pub fn jacobian<T: Real + Scalar + RingCommutative, N: DimName, F: FnMut(T) -> T>(
    data: &mut VectorN<T, N>,
    mut function: F,
) -> MatrixMN<T, N, N>
where
    T: From<<T as approx::AbsDiffEq>::Epsilon>,
    DefaultAllocator: Allocator<T, N, N>,
    DefaultAllocator: Allocator<T, N>,
{
    let len = N::dim();

    let base_eval = apply_new(&data, &mut function);

    let mut data_clone = data.clone();
    let mut final_matrix: MatrixMN<T, N, N> = new_matrix();

    unsafe {
        for index in 0..len {
            *data_clone.get_unchecked_mut(index, 0) += T::from(T::default_epsilon());
            final_matrix.set_column(
                index,
                &((apply_new(&data_clone, &mut function) - &base_eval)
                    / T::from(T::default_epsilon())),
            );
            *data_clone.get_unchecked_mut(index, 0) = *data.get_unchecked_mut(index, 0);
        }
    }
    final_matrix
}

#[inline]
/// TODO: This could be wrong. I dont know how to do partial derivatives
pub fn jacobian_square<
    T: Real + Scalar + RingCommutative,
    R: DimName,
    C: DimName,
    F: FnMut(&T) -> T,
>(
    data: &mut VectorN<T, C>,
    function: &mut Vec<F>,
) -> MatrixMN<T, R, C>
where
    T: From<<T as approx::AbsDiffEq>::Epsilon>,
    DefaultAllocator: Allocator<T, R, C>,
    DefaultAllocator: Allocator<T, C>,
{
    let mut final_matrix: MatrixMN<T, R, C> = new_matrix();
    let epsilon = T::from(T::default_epsilon());

    unsafe {
        for column in 0..C::dim() {
            for row in 0..function.len() {
                let data_point = *data.get_unchecked(row, 0);
                *final_matrix.get_unchecked_mut(row, column) =
                    ((&mut function[column])(&(data_point + epsilon))
                        - (&mut function[column])(&data_point))
                        / epsilon;
            }
        }
    }
    final_matrix
}

#[inline]
pub fn covariance<T: Real>(a: Vec<T>, b: Vec<T>) -> T {
    let len = a.len().min(b.len());
    let mut mean_a = T::zero();
    let mut mean_b = T::zero();

    for index in 0..len {
        mean_a += a[index];
        mean_b += b[index];
    }

    mean_a /= T::from_usize(len).expect("Unable to convert T to usize!");
    mean_b /= T::from_usize(len).expect("Unable to convert T to usize!");

    let mut covariance = T::zero();

    for index in 0..len {
        covariance += (a[index] - mean_a) * (b[index] - mean_b);
    }

    match T::from_usize(len) {
        Some(v) => covariance / v,
        None => T::zero(),
    }
}
