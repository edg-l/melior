//! Blocks.

mod argument;

pub use self::argument::Argument;
use super::{Location, Operation, Type, TypeLike, Value};
use crate::mlir_sys::{
    mlirBlockAddArgument, mlirBlockAppendOwnedOperation, mlirBlockCreate, mlirBlockDestroy,
    mlirBlockEqual, mlirBlockGetArgument, mlirBlockGetFirstOperation, mlirBlockGetNumArguments,
    mlirBlockGetTerminator, mlirBlockInsertOwnedOperation, mlirBlockInsertOwnedOperationAfter,
    mlirBlockInsertOwnedOperationBefore, mlirBlockPrint, mlirOperationEqual,
    mlirOperationGetNextInBlock, MlirBlock,
};
use crate::{
    utility::{into_raw_array, print_callback},
    Error,
};
use std::cell::RefCell;
use std::rc::Rc;
use std::{
    ffi::c_void,
    fmt::{self, Debug, Display, Formatter},
};

/// A block.
pub struct Block {
    raw: MlirBlock,
    pub(crate) owned: bool,
    operations: Vec<Rc<RefCell<Operation>>>,
}

impl Block {
    /// Creates a block.
    pub fn new(arguments: &[(Type, Location)]) -> Self {
        unsafe {
            Self::from_raw(
                mlirBlockCreate(
                    arguments.len() as isize,
                    into_raw_array(
                        arguments
                            .iter()
                            .map(|(argument, _)| argument.to_raw())
                            .collect(),
                    ),
                    into_raw_array(
                        arguments
                            .iter()
                            .map(|(_, location)| location.to_raw())
                            .collect(),
                    ),
                ),
                true,
            )
        }
    }

    /// Gets an argument at a position.
    pub fn argument(&self, position: usize) -> Result<Argument, Error> {
        unsafe {
            if position < self.argument_count() {
                Ok(Argument::from_raw(mlirBlockGetArgument(
                    self.raw,
                    position as isize,
                )))
            } else {
                Err(Error::BlockArgumentPosition(self.to_string(), position))
            }
        }
    }

    /// Gets a number of arguments.
    pub fn argument_count(&self) -> usize {
        unsafe { mlirBlockGetNumArguments(self.raw) as usize }
    }

    /// Gets the first operation.
    pub fn first_operation(&self) -> Option<Rc<RefCell<Operation>>> {
        self.operations.first().cloned()
    }

    /// Gets a terminator operation.
    pub fn terminator(&self) -> Option<Rc<RefCell<Operation>>> {
        let term_op = unsafe { mlirBlockGetTerminator(self.raw) };
        self.operations
            .iter()
            .find(|&op| unsafe { mlirOperationEqual(term_op, RefCell::borrow(op).raw) })
            .cloned()
    }

    /// Gets a parent operation.
    /*
    pub fn parent_operation(&self) -> Option<OperationRef> {
        unsafe { OperationRef::from_option_raw(mlirBlockGetParentOperation(self.raw)) }
    }
    */

    /// Adds an argument.
    pub fn add_argument(&self, r#type: Type, location: Location) -> Value {
        unsafe {
            Value::from_raw(mlirBlockAddArgument(
                self.raw,
                r#type.to_raw(),
                location.to_raw(),
            ))
        }
    }

    /// Appends an operation.
    pub fn append_operation(&mut self, mut operation: Operation) -> Rc<RefCell<Operation>> {
        unsafe {
            mlirBlockAppendOwnedOperation(self.raw, operation.raw);
        }
        operation.owned = false;
        self.operations.push(Rc::new(RefCell::new(operation)));
        self.operations.last().unwrap().clone()
    }

    /// Inserts an operation.
    pub fn insert_operation(
        &mut self,
        position: usize,
        mut operation: Operation,
    ) -> Rc<RefCell<Operation>> {
        unsafe {
            mlirBlockInsertOwnedOperation(self.raw, position as isize, operation.raw);
        }
        operation.owned = false;
        self.operations
            .insert(position, Rc::new(RefCell::new(operation)));
        self.operations[position].clone()
    }

    /// Inserts an operation after another.
    pub fn insert_operation_after(
        &mut self,
        reference: &Operation,
        mut other: Operation,
    ) -> Result<Rc<RefCell<Operation>>, Error> {
        for (i, b) in self.operations.iter().enumerate() {
            if *RefCell::borrow(b) == *reference {
                unsafe {
                    mlirBlockInsertOwnedOperationAfter(self.raw, reference.raw, other.raw);
                }
                other.owned = false;
                self.operations.insert(i + 1, Rc::new(RefCell::new(other)));
                return Ok(self.operations[i].clone());
            }
        }
        Err(Error::OperationNotFound)
    }

    /// Inserts an operation before another.
    pub fn insert_operation_before(
        &mut self,
        reference: &Operation,
        mut other: Operation,
    ) -> Result<Rc<RefCell<Operation>>, Error> {
        for (i, b) in self.operations.iter().enumerate() {
            if *RefCell::borrow(b) == *reference {
                unsafe {
                    mlirBlockInsertOwnedOperationBefore(self.raw, reference.raw, other.raw);
                }
                other.owned = false;
                self.operations.insert(i, Rc::new(RefCell::new(other)));
                return Ok(self.operations[i].clone());
            }
        }
        Err(Error::OperationNotFound)
    }

    /// Detaches a block from a region and assumes its ownership.
    ///
    /// # Safety
    ///
    /// This function might invalidate existing references to the block if you
    /// drop it too early.
    // TODO Implement this for BlockRefMut instead and mark it safe.
    // todo: implñement this in region
    /*
    pub unsafe fn detach(&self) -> Option<Block> {
        if self.parent_region().is_some() {
            mlirBlockDetach(self.raw);

            Some(Block::from_raw(self.raw))
        } else {
            None
        }
    }
    */

    /// Gets a next block in a region.
    // pub fn next_in_region(&self) -> Option<BlockRef> {
    //    unsafe { BlockRef::from_option_raw(mlirBlockGetNextInRegion(self.raw)) }
    // }

    pub(crate) unsafe fn from_raw(raw: MlirBlock, owned: bool) -> Self {
        let mut operations = Vec::default();

        let mut current_op_raw = unsafe { mlirBlockGetFirstOperation(raw) };

        while !current_op_raw.ptr.is_null() {
            let op = unsafe { Operation::from_raw(current_op_raw, false) };
            operations.push(Rc::new(RefCell::new(op)));
            current_op_raw = unsafe { mlirOperationGetNextInBlock(current_op_raw) };
        }

        Self {
            raw,
            owned,
            operations,
        }
    }

    pub(crate) const unsafe fn to_raw(&self) -> MlirBlock {
        self.raw
    }
}

impl Drop for Block {
    fn drop(&mut self) {
        if self.owned {
            unsafe { mlirBlockDestroy(self.raw) };
        }
    }
}

impl PartialEq for Block {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirBlockEqual(self.raw, other.raw) }
    }
}

impl Eq for Block {}

impl Display for Block {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        let mut data = (formatter, Ok(()));

        unsafe {
            mlirBlockPrint(
                self.raw,
                Some(print_callback),
                &mut data as *mut _ as *mut c_void,
            );
        }

        data.1
    }
}

impl Debug for Block {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        writeln!(formatter, "Block(")?;
        Display::fmt(self, formatter)?;
        write!(formatter, ")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dialect::{self, Registry},
        ir::{operation, NamedAttribute, ValueLike},
        utility::register_all_dialects,
        Context,
    };
    use pretty_assertions::assert_eq;

    #[test]
    fn new() {
        Block::new(&[]);
    }

    #[test]
    fn argument() {
        let context = Context::new();
        let r#type = Type::integer(&context, 64);

        assert_eq!(
            Block::new(&[(r#type, Location::unknown(&context))])
                .argument(0)
                .unwrap()
                .r#type(),
            r#type
        );
    }

    #[test]
    fn argument_error() {
        assert_eq!(
            Block::new(&[]).argument(0).unwrap_err(),
            Error::BlockArgumentPosition("<<UNLINKED BLOCK>>\n".into(), 0)
        );
    }

    #[test]
    fn argument_count() {
        assert_eq!(Block::new(&[]).argument_count(), 0);
    }

    /*
    #[test]
    fn parent_region() {
        let region = Region::new();
        let block = region.append_block(Block::new(&[]));

        assert_eq!(block.parent_region().as_deref(), Some(&region));
    }

    #[test]
    fn parent_region_none() {
        let block = Block::new(&[]);

        assert_eq!(block.parent_region(), None);
    }

    #[test]
    fn parent_operation() {
        let context = Context::new();
        let module = Module::new(Location::unknown(&context));

        assert_eq!(
            module.body().parent_operation(),
            Some(module.as_operation())
        );
    }

    #[test]
    fn parent_operation_none() {
        let block = Block::new(&[]);

        assert_eq!(block.parent_operation(), None);
    }
    */

    #[test]
    fn terminator() {
        let registry = dialect::Registry::new();
        register_all_dialects(&registry);

        let context = Context::new();
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();

        let mut block = Block::new(&[]);

        let operation = block.append_operation(
            operation::Builder::new("func.return", Location::unknown(&context)).build(),
        );

        assert_eq!(block.terminator(), Some(operation));
    }

    #[test]
    fn terminator_none() {
        assert_eq!(Block::new(&[]).terminator(), None);
    }

    #[test]
    fn first_operation() {
        let context = Context::new();
        let registry = Registry::new();
        register_all_dialects(&registry);
        context.append_dialect_registry(&registry);
        context.get_or_load_dialect("arith");
        let mut block = Block::new(&[]);

        let op = block.append_operation(
            operation::Builder::new("arith.constant", Location::unknown(&context))
                .add_results(&[Type::integer(&context, 32)])
                .add_attributes(
                    &[NamedAttribute::new_parsed(&context, "value", "0 : i32").unwrap()],
                )
                .build(),
        );

        assert_eq!(block.first_operation(), Some(op));
    }

    #[test]
    fn first_operation_none() {
        let block = Block::new(&[]);

        assert_eq!(block.first_operation(), None);
    }

    #[test]
    fn append_operation() {
        let context = Context::new();
        let registry = Registry::new();
        register_all_dialects(&registry);
        context.append_dialect_registry(&registry);
        context.get_or_load_dialect("arith");
        let mut block = Block::new(&[]);

        block.append_operation(
            operation::Builder::new("arith.constant", Location::unknown(&context))
                .add_results(&[Type::integer(&context, 32)])
                .add_attributes(
                    &[NamedAttribute::new_parsed(&context, "value", "0 : i32").unwrap()],
                )
                .build(),
        );
    }

    #[test]
    fn insert_operation() {
        let context = Context::new();
        let registry = Registry::new();
        register_all_dialects(&registry);
        context.append_dialect_registry(&registry);
        context.get_or_load_dialect("arith");
        let mut block = Block::new(&[]);

        block.insert_operation(
            0,
            operation::Builder::new("arith.constant", Location::unknown(&context))
                .add_results(&[Type::integer(&context, 32)])
                .add_attributes(
                    &[NamedAttribute::new_parsed(&context, "value", "0 : i32").unwrap()],
                )
                .build(),
        );
    }

    #[test]
    fn insert_operation_after() {
        let context = Context::new();
        let registry = Registry::new();
        register_all_dialects(&registry);
        context.append_dialect_registry(&registry);
        context.get_or_load_dialect("arith");
        let mut block = Block::new(&[]);

        let first_operation = block.append_operation(
            operation::Builder::new("arith.constant", Location::unknown(&context))
                .add_results(&[Type::integer(&context, 32)])
                .add_attributes(
                    &[NamedAttribute::new_parsed(&context, "value", "0 : i32").unwrap()],
                )
                .build(),
        );
        let second_operation = block
            .insert_operation_after(
                &first_operation.borrow(),
                operation::Builder::new("arith.constant", Location::unknown(&context))
                    .add_results(&[Type::integer(&context, 32)])
                    .add_attributes(&[
                        NamedAttribute::new_parsed(&context, "value", "0 : i32").unwrap()
                    ])
                    .build(),
            )
            .unwrap();

        assert_eq!(block.first_operation(), Some(first_operation));
        assert_eq!(block.operations.get(1), Some(&second_operation));
    }

    #[test]
    fn insert_operation_before() {
        let context = Context::new();
        let registry = Registry::new();
        register_all_dialects(&registry);
        context.append_dialect_registry(&registry);
        context.get_or_load_dialect("arith");
        let mut block = Block::new(&[]);

        let second_operation = block.append_operation(
            operation::Builder::new("arith.constant", Location::unknown(&context))
                .add_results(&[Type::integer(&context, 32)])
                .add_attributes(
                    &[NamedAttribute::new_parsed(&context, "value", "0 : i32").unwrap()],
                )
                .build(),
        );
        let first_operation = block
            .insert_operation_before(
                &second_operation.borrow(),
                operation::Builder::new("arith.constant", Location::unknown(&context))
                    .add_results(&[Type::integer(&context, 32)])
                    .add_attributes(&[
                        NamedAttribute::new_parsed(&context, "value", "0 : i32").unwrap()
                    ])
                    .build(),
            )
            .unwrap();

        assert_eq!(block.operations.first(), Some(&first_operation));
        assert_eq!(block.operations.get(1), Some(&second_operation));
    }

    /*
    #[test]
    fn detach() {
        let region = Region::new();
        let block = region.append_block(Block::new(&[]));

        assert_eq!(
            unsafe { block.detach() }.unwrap().to_string(),
            "<<UNLINKED BLOCK>>\n"
        );
    }
    */

    #[test]
    fn display() {
        assert_eq!(Block::new(&[]).to_string(), "<<UNLINKED BLOCK>>\n");
    }

    #[test]
    fn debug() {
        assert_eq!(
            format!("{:?}", &Block::new(&[])),
            "Block(\n<<UNLINKED BLOCK>>\n)"
        );
    }
}
