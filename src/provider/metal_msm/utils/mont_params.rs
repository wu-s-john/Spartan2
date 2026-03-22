use num_bigint::{BigInt, BigUint, Sign};

pub fn calc_nsafe(log_limb_size: u32) -> usize {
    let max_int_width = 32;
    let rhs = 2u64.pow(max_int_width);
    let mut k = 1usize;
    let x = 2u64.pow(2u32 * log_limb_size);
    while (k as u64) * x <= rhs {
        k += 1;
    }
    k / 2
}

pub fn calc_mont_radix(num_limbs: usize, log_limb_size: u32) -> BigUint {
    BigUint::from(2u32).pow(num_limbs as u32 * log_limb_size)
}

fn egcd(a: &BigInt, b: &BigInt) -> (BigInt, BigInt, BigInt) {
    if *a == BigInt::from(0u32) {
        return (b.clone(), BigInt::from(0u32), BigInt::from(1u32));
    }
    let (g, x, y) = egcd(&(b % a), a);
    (g, y - (b / a) * x.clone(), x.clone())
}

pub fn calc_inv_and_pprime(p: &BigUint, r: &BigUint) -> (BigUint, BigUint) {
    assert!(*r != BigUint::from(0u32));

    let p_bigint = BigInt::from_biguint(Sign::Plus, p.clone());
    let r_bigint = BigInt::from_biguint(Sign::Plus, r.clone());
    let one = BigInt::from(1u32);
    let (_, mut rinv, mut pprime) = egcd(
        &BigInt::from_biguint(Sign::Plus, r.clone()),
        &BigInt::from_biguint(Sign::Plus, p.clone()),
    );

    if rinv.sign() == Sign::Minus {
        rinv = BigInt::from_biguint(Sign::Plus, p.clone()) + rinv;
    }
    if pprime.sign() == Sign::Minus {
        pprime = BigInt::from_biguint(Sign::Plus, r.clone()) + pprime;
    }

    assert!(
        (BigInt::from_biguint(Sign::Plus, r.clone()) * &rinv % &p_bigint)
            - (&p_bigint * &pprime % &p_bigint)
            == one
    );
    assert!((BigInt::from_biguint(Sign::Plus, r.clone()) * &rinv % &p_bigint) == one);
    assert!((&p_bigint * &pprime % &r_bigint) == one);

    (rinv.to_biguint().unwrap(), pprime.to_biguint().unwrap())
}

pub fn calc_rinv_and_n0(p: &BigUint, r: &BigUint, log_limb_size: u32) -> (BigUint, u32) {
    let (rinv, pprime) = calc_inv_and_pprime(p, r);
    let pprime = BigInt::from_biguint(Sign::Plus, pprime);
    let neg_n_inv = BigInt::from_biguint(Sign::Plus, r.clone()) - pprime;
    let n0 = neg_n_inv % BigInt::from(2u32.pow(log_limb_size));
    let n0 = n0.to_biguint().unwrap().to_u32_digits()[0];
    (rinv, n0)
}
