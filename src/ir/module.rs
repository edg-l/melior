use super::{Block, Location, Operation};
use crate::mlir_sys::{
    mlirModuleCreateEmpty, mlirModuleCreateParse, mlirModuleDestroy, mlirModuleFromOperation,
    mlirModuleGetBody, mlirModuleGetContext, mlirModuleGetOperation, MlirModule,
};
use crate::{
    context::{Context, ContextRef},
    string_ref::StringRef,
};
use std::marker::PhantomData;

/// A module.
#[derive(Debug)]
pub struct Module<'c> {
    raw: MlirModule,
    pub operation: Operation,
    pub body: Block,
    _context: PhantomData<&'c Context>,
}

impl<'c> Module<'c> {
    /// Creates a module.
    pub fn new(location: Location) -> Self {
        unsafe { Self::from_raw(mlirModuleCreateEmpty(location.to_raw())) }
    }

    /// Parses a module.
    pub fn parse(context: &Context, source: &str) -> Option<Self> {
        // TODO Should we allocate StringRef locally because sources can be big?
        unsafe {
            Self::from_option_raw(mlirModuleCreateParse(
                context.to_raw(),
                StringRef::from(source).to_raw(),
            ))
        }
    }

    /// Gets a context.
    pub fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirModuleGetContext(self.raw)) }
    }

    /// Converts an operation into a module.
    pub fn from_operation(operation: Operation) -> Option<Self> {
        unsafe { Self::from_option_raw(mlirModuleFromOperation(operation.raw)) }
    }

    unsafe fn from_raw(raw: MlirModule) -> Self {
        let operation = unsafe { Operation::from_raw(mlirModuleGetOperation(raw), false) };
        let body = unsafe { Block::from_raw(mlirModuleGetBody(raw), false) };
        Self {
            raw,
            operation,
            body,
            _context: Default::default(),
        }
    }

    unsafe fn from_option_raw(raw: MlirModule) -> Option<Self> {
        if raw.ptr.is_null() {
            None
        } else {
            Some(Self::from_raw(raw))
        }
    }

    pub(crate) const unsafe fn to_raw(&self) -> MlirModule {
        self.raw
    }
}

impl<'c> Drop for Module<'c> {
    fn drop(&mut self) {
        unsafe { mlirModuleDestroy(self.raw) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dialect::Registry,
        ir::{operation, Block, Region},
        utility::register_all_dialects,
    };

    #[test]
    fn new() {
        Module::new(Location::new(&Context::new(), "foo", 42, 42));
    }

    #[test]
    fn context() {
        Module::new(Location::new(&Context::new(), "foo", 42, 42)).context();
    }

    #[test]
    fn parse() {
        assert!(Module::parse(&Context::new(), "module{}").is_some());
    }

    #[test]
    fn parse_none() {
        assert!(Module::parse(&Context::new(), "module{").is_none());
    }

    #[test]
    fn from_operation() {
        let context = Context::new();

        let mut region = Region::new();
        region.append_block(Block::new(&[]));

        let module = Module::from_operation(
            operation::Builder::new("builtin.module", Location::unknown(&context))
                .add_regions(vec![region])
                .build(),
        )
        .unwrap();

        assert!(module.operation.verify());
        assert_eq!(module.operation.to_string(), "module {\n}\n")
    }

    #[test]
    fn from_operation_fail() {
        let context = Context::new();
        let registry = Registry::new();
        register_all_dialects(&registry);
        context.append_dialect_registry(&registry);
        context.get_or_load_dialect("func");

        assert!(Module::from_operation(
            operation::Builder::new("func.func", Location::unknown(&context)).build()
        )
        .is_none());
    }
}
