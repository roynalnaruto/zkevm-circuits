use crate::arith_helpers::*;
use crate::common::ROUND_CONSTANTS;
use crate::keccak_arith::KeccakFArith;
use halo2::circuit::Cell;
use halo2::plonk::Instance;
use halo2::{
    circuit::Region,
    plonk::{
        Advice, Column, ConstraintSystem, Error, Expression, VirtualCells,
    },
    poly::Rotation,
};
use pasta_curves::arithmetic::FieldExt;
use std::marker::PhantomData;

#[derive(Clone, Debug)]
pub struct IotaB9Config<F> {
    q_enable: Expression<F>,
    state: [Column<Advice>; 25],
    pub(crate) round_ctant_b9: Column<Advice>,
    pub(crate) round_constants: Column<Instance>,
    _marker: PhantomData<F>,
}

impl<F: FieldExt> IotaB9Config<F> {
    // We assume state is recieved in base-9.
    pub fn configure(
        q_enable_fn: impl FnOnce(&mut VirtualCells<'_, F>) -> Expression<F>,
        meta: &mut ConstraintSystem<F>,
        state: [Column<Advice>; 25],
        round_ctant_b9: Column<Advice>,
        round_constants: Column<Instance>,
    ) -> IotaB9Config<F> {
        let mut q_enable = Expression::Constant(F::zero());
        // Enable copy constraints over PI and the Advices.
        meta.enable_equality(round_ctant_b9.into());
        meta.enable_equality(round_constants.into());
        meta.create_gate("iota_b9", |meta| {
            // def iota_b9(state: List[List[int], round_constant_base9: int):
            //     d = round_constant_base9
            //     # state[0][0] has 2*a + b + 3*c already, now add 2*d to make it 2*a + b + 3*c + 2*d
            //     # coefficient in 0~8
            //     state[0][0] += 2*d
            //     return state
            q_enable = q_enable_fn(meta);
            let state_00 = meta.query_advice(state[0], Rotation::cur())
                + (Expression::Constant(F::from(2))
                    * meta.query_advice(round_ctant_b9, Rotation::cur()));
            let next_lane = meta.query_advice(state[0], Rotation::next());
            vec![q_enable.clone() * (state_00 - next_lane)]
        });
        IotaB9Config {
            q_enable,
            state,
            round_ctant_b9,
            round_constants,
            _marker: PhantomData,
        }
    }

    // We need to enable q_enable outside in parallel to the call to this!
    pub fn assign_state_and_rc(
        &self,
        region: &mut Region<'_, F>,
        offset: usize,
        state: [F; 25],
        round: usize,
        absolute_row: usize,
    ) -> Result<([F; 25], usize), Error> {
        for (idx, lane) in state.iter().enumerate() {
            region.assign_advice(
                || format!("assign state {}", idx),
                self.state[idx],
                offset,
                || Ok(*lane),
            )?;
        }

        self.assign_round_ctant_b9(region, offset, absolute_row)?;

        // Apply iota_b9 outside circuit
        let out_state = KeccakFArith::iota_b9(
            &state_to_biguint(state),
            ROUND_CONSTANTS[round],
        );
        let out_state = state_bigint_to_pallas(out_state);

        for (idx, lane) in out_state.iter().enumerate() {
            region.assign_advice(
                || format!("assign state {}", idx),
                self.state[idx],
                offset + 1,
                || Ok(*lane),
            )?;
        }
        Ok((out_state, offset + 1))
    }

    // We need to enable q_enable outside in parallel to the call to this!
    pub fn copy_state_flag_and_assing_rc(
        &self,
        region: &mut Region<'_, F>,
        offset: usize,
        state: [(Cell, F); 25],
        round: usize,
        absolute_row: usize,
        flag: (Cell, F),
    ) -> Result<([F; 25], usize), Error> {
        let mut state_array = [F::zero(); 25];
        for (idx, (cell, value)) in state.iter().enumerate() {
            // Copy value into state_array
            state_array[idx] = *value;
            let new_cell = region.assign_advice(
                || format!("assign state {}", idx),
                self.state[idx],
                offset,
                || Ok(*value),
            )?;

            region.constrain_equal(*cell, new_cell)?;
        }

        self.assign_round_ctant_b9(region, offset, absolute_row)?;

        let offset = self.copy_flag(region, offset, flag)?;

        // Apply iota_b9 outside circuit
        let out_state = KeccakFArith::iota_b9(
            &state_to_biguint(state_array),
            ROUND_CONSTANTS[round],
        );
        let out_state = state_bigint_to_pallas(out_state);

        for (idx, lane) in out_state.iter().enumerate() {
            region.assign_advice(
                || format!("assign state {}", idx),
                self.state[idx],
                offset,
                || Ok(*lane),
            )?;
        }
        Ok((out_state, offset))
    }

    /// Assigns the `is_mixing` flag to the `round_ctant_b9` Advice column at `Rotation::next` (offset + 1)
    fn copy_flag(
        &self,
        region: &mut Region<'_, F>,
        offset: usize,
        flag: (Cell, F),
    ) -> Result<usize, Error> {
        let obtained_cell = region.assign_advice(
            || format!("assign is_mixing flag {:?}", flag.1),
            self.round_ctant_b9,
            offset,
            || Ok(flag.1),
        )?;
        region.constrain_equal(flag.0, obtained_cell)?;

        Ok(1)
    }

    /// Assigns the ROUND_CONSTANTS_BASE_9 to the `absolute_row` passed asn an absolute instance column.
    /// Returns the new offset after the assigment.
    fn assign_round_ctant_b9(
        &self,
        region: &mut Region<'_, F>,
        offset: usize,
        absolute_row: usize,
    ) -> Result<usize, Error> {
        region.assign_advice_from_instance(
            // `absolute_row` is the absolute offset in the overall Region where the Column is laying.
            || format!("assign round_ctant_b9 {}", absolute_row),
            self.round_constants,
            absolute_row,
            self.round_ctant_b9,
            offset,
        )?;

        Ok(offset + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::*;
    use crate::keccak_arith::*;
    use halo2::circuit::Layouter;
    use halo2::plonk::{Advice, Column, ConstraintSystem, Error};
    use halo2::{circuit::SimpleFloorPlanner, dev::MockProver, plonk::Circuit};
    use itertools::Itertools;
    use pasta_curves::arithmetic::FieldExt;
    use pasta_curves::pallas;
    use std::convert::TryInto;
    use std::marker::PhantomData;

    #[test]
    fn test_iota_b9_gate_with_flag() {
        #[derive(Default)]
        struct MyCircuit<F> {
            in_state: [F; 25],
            // This usize is indeed pointing the exact row of the ROUND_CTANTS_B9 we want to use.
            flag: bool,
            round: usize,
            _marker: PhantomData<F>,
        }

        impl<F: FieldExt> Circuit<F> for MyCircuit<F> {
            type Config = IotaB9Config<F>;
            type FloorPlanner = SimpleFloorPlanner;

            fn without_witnesses(&self) -> Self {
                Self::default()
            }

            fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
                let state: [Column<Advice>; 25] = (0..25)
                    .map(|_| {
                        let column = meta.advice_column();
                        meta.enable_equality(column.into());
                        column
                    })
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();
                let round_ctant_b9 = meta.advice_column();
                // Enable equalty
                meta.enable_equality(round_ctant_b9.into());
                // Allocate space for the round constants in base-9 which is an instance column
                let round_ctants = meta.instance_column();
                meta.enable_equality(round_ctants.into());

                // Since we're not using a selector and want to test IotaB9 with the Mixing step, we make q_enable query
                // the round_ctant_b9 at `Rotation::next`.
                IotaB9Config::configure(
                    |meta| meta.query_advice(round_ctant_b9, Rotation::next()),
                    meta,
                    state,
                    round_ctant_b9,
                    round_ctants,
                )
            }

            fn synthesize(
                &self,
                config: Self::Config,
                mut layouter: impl Layouter<F>,
            ) -> Result<(), Error> {
                let offset: usize = 0;

                // Witness mixing_flag at offset = 1
                let val: F = self.flag.into();
                let flag: (Cell, F) = layouter.assign_region(
                    || "witness_is_mixing_flag",
                    |mut region| {
                        let offset = 1;
                        let cell = region.assign_advice(
                            || "assign is_mising",
                            config.round_ctant_b9,
                            offset,
                            || Ok(val),
                        )?;
                        Ok((cell, val))
                    },
                )?;

                // Witness `input_state` and get the Cells back at offset = 0
                let in_state: [(Cell, F); 25] = layouter.assign_region(
                    || "Witness input state",
                    |mut region| {
                        let mut state: Vec<(Cell, F)> = Vec::with_capacity(25);
                        for (idx, val) in self.in_state.iter().enumerate() {
                            let cell = region.assign_advice(
                                || "witness input state",
                                config.state[idx],
                                offset,
                                || Ok(*val),
                            )?;
                            state.push((cell, *val))
                        }

                        Ok(state.try_into().unwrap())
                    },
                )?;

                // Assign `input_state`, `flag` and round_ctant.
                layouter.assign_region(
                    || "assign input & output state + flag",
                    |mut region| {
                        let (_, offset) = config
                            .copy_state_flag_and_assing_rc(
                                &mut region,
                                offset,
                                in_state,
                                self.round,
                                // Abs row is 0 since B9 PI are allocated at 0idx
                                0,
                                flag,
                            )?;
                        Ok(())
                    },
                )
            }
        }

        let input1: State = [
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ];
        let mut in_biguint = StateBigInt::default();
        let mut in_state: [pallas::Base; 25] = [pallas::Base::zero(); 25];

        for (x, y) in (0..5).cartesian_product(0..5) {
            in_biguint[(x, y)] = convert_b2_to_b9(input1[x][y]);
            in_state[5 * x + y] = big_uint_to_pallas(&in_biguint[(x, y)]);
        }

        // Define the round we're going to run.
        let round = 2;
        let s1_arith =
            KeccakFArith::iota_b9(&in_biguint, ROUND_CONSTANTS[round]);

        let circuit = MyCircuit::<pallas::Base> {
            in_state,
            flag: true,
            round,
            _marker: PhantomData,
        };

        let constants: Vec<pallas::Base> = ROUND_CONSTANTS
            .iter()
            .map(|num| big_uint_to_pallas(&convert_b2_to_b9(*num)))
            .collect();

        let prover =
            MockProver::<pallas::Base>::run(9, &circuit, vec![constants])
                .unwrap();

        assert_eq!(prover.verify(), Ok(()));
    }
}
