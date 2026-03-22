use num_bigint::BigUint;

pub fn calc_barrett_mu(p: &BigUint) -> BigUint {
    let k = p.bits() as u32;
    let numerator = BigUint::from(1u32) << (2 * k);
    numerator / p
}
