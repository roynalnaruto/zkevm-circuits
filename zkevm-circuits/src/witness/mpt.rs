use crate::{
    evm_circuit::witness::Rw,
    table::{AccountFieldTag, MPTProofType},
    util::word,
};
use eth_types::{Address, Field, ToScalar, Word};
use halo2_proofs::circuit::Value;
use itertools::Itertools;
use std::collections::BTreeMap;

/// An MPT update whose validity is proved by the MptCircuit
#[derive(Debug, Clone, Copy)]
pub struct MptUpdate {
    key: Key,
    old_value: Word,
    new_value: Word,
    old_root: Word,
    new_root: Word,
}

impl MptUpdate {
    fn proof_type<F: Field>(&self) -> F {
        let proof_type = match self.key {
            Key::AccountStorage { .. } => {
                if self.old_value.is_zero() && self.new_value.is_zero() {
                    MPTProofType::NonExistingStorageProof
                } else {
                    MPTProofType::StorageMod
                }
            }
            Key::Account { field_tag, .. } => field_tag.into(),
        };
        F::from(proof_type as u64)
    }
}

/// All the MPT updates in the MptCircuit, accessible by their key
#[derive(Default, Clone, Debug)]
pub struct MptUpdates {
    old_root: Word,
    updates: BTreeMap<Key, MptUpdate>,
}

/// The field element encoding of an MPT update, which is used by the MptTable
#[derive(Debug, Clone, Copy)]
pub struct MptUpdateRow<F>(pub(crate) [F; 12]);

impl MptUpdates {
    pub(crate) fn old_root(&self) -> Word {
        self.old_root
    }

    pub(crate) fn get(&self, row: &Rw) -> Option<MptUpdate> {
        key(row).map(|key| *self.updates.get(&key).expect("missing key in mpt updates"))
    }

    pub(crate) fn mock_from(rows: &[Rw]) -> Self {
        let mock_old_root = Word::from(0xcafeu64);
        let map: BTreeMap<_, _> = rows
            .iter()
            .group_by(|row| key(row))
            .into_iter()
            .filter_map(|(key, rows)| key.map(|key| (key, rows)))
            .enumerate()
            .map(|(i, (key, mut rows))| {
                let first = rows.next().unwrap();
                let last = rows.last().unwrap_or(first);
                let key_exists = key;
                let key = key.set_non_exists(value_prev(first), value(last));
                (
                    key_exists,
                    MptUpdate {
                        key,
                        old_root: Word::from(i as u64) + mock_old_root,
                        new_root: Word::from(i as u64 + 1) + mock_old_root,
                        old_value: value_prev(first),
                        new_value: value(last),
                    },
                )
            })
            .collect();
        MptUpdates {
            updates: map,
            old_root: mock_old_root,
        }
    }

    pub(crate) fn table_assignments<F: Field>(&self) -> Vec<MptUpdateRow<Value<F>>> {
        self.updates
            .values()
            .map(|update| {
                let (new_root, old_root) = update.root_assignments();
                let (new_value, old_value) = update.value_assignments();
                let (storage_key_lo, storage_key_hi) =
                    word::Word::<F>::from(update.key.storage_key()).into_lo_hi();
                let (new_root_lo, new_root_hi) = word::Word::<F>::from(new_root).into_lo_hi();
                let (old_root_lo, old_root_hi) = word::Word::<F>::from(old_root).into_lo_hi();
                let (new_value_lo, new_value_hi) = word::Word::<F>::from(new_value).into_lo_hi();
                let (old_value_lo, old_value_hi) = word::Word::<F>::from(old_value).into_lo_hi();
                let address = update.key.address().to_scalar().unwrap();

                MptUpdateRow(
                    [
                        address,
                        storage_key_lo,
                        storage_key_hi,
                        update.proof_type(),
                        new_root_lo,
                        new_root_hi,
                        old_root_lo,
                        old_root_hi,
                        new_value_lo,
                        new_value_hi,
                        old_value_lo,
                        old_value_hi,
                    ]
                    .map(|v| Value::known(v)),
                )
            })
            .collect()
    }
}

impl MptUpdate {
    pub(crate) fn value_assignments(&self) -> (Word, Word) {
        (self.new_value, self.old_value)
    }
    pub(crate) fn root_assignments(&self) -> (Word, Word) {
        (self.new_root, self.old_root)
    }
}

#[derive(Eq, PartialEq, Hash, Clone, Debug, Copy, PartialOrd, Ord)]
enum Key {
    Account {
        address: Address,
        field_tag: AccountFieldTag,
    },
    AccountStorage {
        tx_id: usize,
        address: Address,
        storage_key: Word,
        exists: bool,
    },
}

impl Key {
    // If the transition is Storage 0 -> 0, set the key as non-existing storage.
    // If the transition is CodeHash 0 -> 0, set the key as non-existing account.
    // Otherwise return the key unmodified.
    fn set_non_exists(self, value_prev: Word, value: Word) -> Self {
        if value_prev.is_zero() && value.is_zero() {
            match self {
                Key::Account { address, field_tag } => {
                    if matches!(field_tag, AccountFieldTag::CodeHash) {
                        Key::Account {
                            address,
                            field_tag: AccountFieldTag::NonExisting,
                        }
                    } else {
                        self
                    }
                }
                Key::AccountStorage {
                    tx_id,
                    address,
                    storage_key,
                    ..
                } => Key::AccountStorage {
                    tx_id,
                    address,
                    storage_key,
                    exists: false,
                },
            }
        } else {
            self
        }
    }
    fn address(&self) -> Address {
        match self {
            Self::Account { address, .. } | Self::AccountStorage { address, .. } => *address,
        }
    }
    fn storage_key(&self) -> Word {
        match self {
            Self::Account { .. } => Word::zero(),
            Self::AccountStorage { storage_key, .. } => *storage_key,
        }
    }
}

impl<F> MptUpdateRow<F> {
    /// The individual values of the row, in the column order used by the
    /// MptTable
    pub fn values(&self) -> impl Iterator<Item = &F> {
        self.0.iter()
    }
}

fn key(row: &Rw) -> Option<Key> {
    match row {
        Rw::Account {
            account_address,
            field_tag,
            ..
        } => Some(Key::Account {
            address: *account_address,
            field_tag: *field_tag,
        }),
        Rw::AccountStorage {
            tx_id,
            account_address,
            storage_key,
            ..
        } => Some(Key::AccountStorage {
            tx_id: *tx_id,
            address: *account_address,
            storage_key: *storage_key,
            exists: true,
        }),
        _ => None,
    }
}

fn value(row: &Rw) -> Word {
    match row {
        Rw::Account { value, .. } => *value,
        Rw::AccountStorage { value, .. } => *value,
        _ => unreachable!(),
    }
}

fn value_prev(row: &Rw) -> Word {
    match row {
        Rw::Account { value_prev, .. } => *value_prev,
        Rw::AccountStorage { value_prev, .. } => *value_prev,
        _ => unreachable!(),
    }
}
