use bus_mapping::circuit_input_builder::Call;
use eth_types::{evm_types::GasCost, Field, ToScalar};
use gadgets::util::Expr;
use halo2_proofs::{circuit::Value, plonk::Error};

use crate::{
    evm_circuit::{
        execution::ExecutionGadget,
        step::ExecutionState,
        util::{constraint_builder::EVMConstraintBuilder, CachedRegion, Cell},
    },
    table::CallContextFieldTag,
    witness::{Block, ExecStep, Transaction},
};

#[derive(Clone, Debug)]
pub struct IdentityGadget<F> {
    is_success: Cell<F>,
    callee_address: Cell<F>,
    caller_id: Cell<F>,
    call_data_offset: Cell<F>,
    call_data_length: Cell<F>,
    return_data_offset: Cell<F>,
    return_data_length: Cell<F>,
}

impl<F: Field> ExecutionGadget<F> for IdentityGadget<F> {
    const EXECUTION_STATE: ExecutionState = ExecutionState::PrecompileIdentity;

    const NAME: &'static str = "IDENTITY";

    fn configure(cb: &mut EVMConstraintBuilder<F>) -> Self {
        let [is_success, callee_address, caller_id, call_data_offset, call_data_length, return_data_offset, return_data_length] =
            [
                CallContextFieldTag::IsSuccess,
                CallContextFieldTag::CalleeAddress,
                CallContextFieldTag::CallerId,
                CallContextFieldTag::CallDataOffset,
                CallContextFieldTag::CallDataLength,
                CallContextFieldTag::ReturnDataOffset,
                CallContextFieldTag::ReturnDataLength,
            ]
            .map(|tag| cb.call_context(None, tag));

        cb.precompile_info_lookup(
            cb.execution_state().as_u64().expr(),
            callee_address.expr(),
            GasCost::PRECOMPILE_IDENTITY_BASE.expr(),
        );

        Self {
            is_success,
            callee_address,
            caller_id,
            call_data_offset,
            call_data_length,
            return_data_offset,
            return_data_length,
        }
    }

    fn assign_exec_step(
        &self,
        region: &mut CachedRegion<'_, '_, F>,
        offset: usize,
        _block: &Block<F>,
        _tx: &Transaction,
        call: &Call,
        _step: &ExecStep,
    ) -> Result<(), Error> {
        self.is_success.assign(
            region,
            offset,
            Value::known(F::from(u64::from(call.is_success))),
        )?;
        self.callee_address.assign(
            region,
            offset,
            Value::known(call.code_address().unwrap().to_scalar().unwrap()),
        )?;
        self.caller_id.assign(
            region,
            offset,
            Value::known(F::from(call.caller_id.try_into().unwrap())),
        )?;
        self.call_data_offset.assign(
            region,
            offset,
            Value::known(F::from(call.call_data_offset)),
        )?;
        self.call_data_length.assign(
            region,
            offset,
            Value::known(F::from(call.call_data_length)),
        )?;
        self.return_data_offset.assign(
            region,
            offset,
            Value::known(F::from(call.return_data_offset)),
        )?;
        self.return_data_length.assign(
            region,
            offset,
            Value::known(F::from(call.return_data_length)),
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use eth_types::bytecode;
    use mock::TestContext;

    use crate::test_util::CircuitTestBuilder;

    #[test]
    fn precompile_identity_test() {
        let bytecode = bytecode! {
            // place params in memory
            PUSH1(0xff)
            PUSH1(0x00)
            MSTORE
            // do static call to 0x04
            PUSH1(0x01)
            PUSH1(0x3f)
            PUSH1(0x01)
            PUSH1(0x1f)
            PUSH1(0x04)
            PUSH1(0xff)
            STATICCALL
            // put result on the stack
            POP
            PUSH1(0x20)
            MLOAD
        };

        CircuitTestBuilder::new_from_test_ctx(
            TestContext::<2, 1>::simple_ctx_with_bytecode(bytecode).unwrap(),
        )
        .run();
    }
}
