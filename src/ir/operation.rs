//! Operations and operation builders.

mod builder;
mod result;

pub use self::{builder::Builder, result::ResultValue};
use super::{Identifier, Region, Value};
use crate::mlir_sys::{
    mlirOpPrintingFlagsCreate, mlirOpPrintingFlagsEnableDebugInfo, mlirOperationClone,
    mlirOperationDestroy, mlirOperationDump, mlirOperationEqual, mlirOperationGetContext,
    mlirOperationGetFirstRegion, mlirOperationGetName, mlirOperationGetNumResults,
    mlirOperationGetResult, mlirOperationPrintWithFlags, mlirOperationVerify,
    mlirRegionGetNextInOperation, MlirOperation,
};
use crate::utility::print_debug_callback;
use crate::{context::ContextRef, utility::print_callback, Error};
use core::fmt;
use std::{
    ffi::c_void,
    fmt::{Debug, Display, Formatter},
};

/// An operation.
pub struct Operation {
    pub(crate) raw: MlirOperation,
    pub(crate) owned: bool,
    regions: Vec<Region>,
}

impl Operation {
    /// Gets a context.
    pub fn context(&self) -> ContextRef {
        unsafe { ContextRef::from_raw(mlirOperationGetContext(self.raw)) }
    }

    /// Gets a name.
    pub fn name(&self) -> Identifier {
        unsafe { Identifier::from_raw(mlirOperationGetName(self.raw)) }
    }

    /// Gets a result at a position.
    pub fn result(&self, position: usize) -> Result<result::ResultValue, Error> {
        unsafe {
            if position < self.result_count() {
                Ok(result::ResultValue::from_raw(mlirOperationGetResult(
                    self.raw,
                    position as isize,
                )))
            } else {
                Err(Error::OperationResultPosition(self.to_string(), position))
            }
        }
    }

    /// Gets a number of results.
    pub fn result_count(&self) -> usize {
        unsafe { mlirOperationGetNumResults(self.raw) as usize }
    }

    /// Gets a region at index.
    pub fn region(&self, index: usize) -> Option<&Region> {
        self.regions.get(index)
    }

    /// Gets a mutable region at index.
    pub fn region_mut(&mut self, index: usize) -> Option<&mut Region> {
        self.regions.get_mut(index)
    }

    /// Gets the number of regions.
    pub fn region_count(&self) -> usize {
        self.regions.len()
    }

    pub fn debug_print(&self) -> String {
        let mut data = String::new();

        unsafe {
            let flags = mlirOpPrintingFlagsCreate();
            mlirOpPrintingFlagsEnableDebugInfo(flags, true, false);
            mlirOperationPrintWithFlags(
                self.raw,
                flags,
                Some(print_debug_callback),
                &mut data as *mut _ as *mut c_void,
            );
        };

        data
    }

    // Gets the next operation in the same block.
    /*
    pub fn next_in_block(&self) -> Option<OperationRef> {
        unsafe {
            let operation = mlirOperationGetNextInBlock(self.raw);

            if operation.ptr.is_null() {
                None
            } else {
                Some(OperationRef::from_raw(operation))
            }
        }
    }
    */

    /// Verifies an operation.
    pub fn verify(&self) -> bool {
        unsafe { mlirOperationVerify(self.raw) }
    }

    /// Dumps an operation.
    pub fn dump(&self) {
        unsafe { mlirOperationDump(self.raw) }
    }

    /// Gets this operation from the raw handle, population all the regions, recursively.
    pub(crate) unsafe fn from_raw(raw: MlirOperation, owned: bool) -> Self {
        let mut regions = Vec::default();

        let mut current_region_raw = mlirOperationGetFirstRegion(raw);

        while !current_region_raw.ptr.is_null() {
            let region = unsafe { Region::from_raw(current_region_raw, false) };
            regions.push(region);
            current_region_raw = unsafe { mlirRegionGetNextInOperation(current_region_raw) };
        }

        Self {
            raw,
            owned,
            regions,
        }
    }
}

impl Clone for Operation {
    fn clone(&self) -> Self {
        unsafe { Self::from_raw(mlirOperationClone(self.raw), true) }
    }
}

impl Drop for Operation {
    fn drop(&mut self) {
        if self.owned {
            unsafe { mlirOperationDestroy(self.raw) };
        }
    }
}

impl PartialEq for Operation {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirOperationEqual(self.raw, other.raw) }
    }
}

impl Eq for Operation {}

impl Display for Operation {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        let mut data = (formatter, Ok(()));

        unsafe {
            let flags = mlirOpPrintingFlagsCreate();
            mlirOpPrintingFlagsEnableDebugInfo(flags, false, false);
            mlirOperationPrintWithFlags(
                self.raw,
                flags,
                Some(print_callback),
                &mut data as *mut _ as *mut c_void,
            );
        }

        data.1
    }
}

impl Debug for Operation {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        writeln!(formatter, "Operation(")?;
        Display::fmt(self, formatter)?;
        write!(formatter, ")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{context::Context, ir::Location};
    use pretty_assertions::assert_eq;

    #[test]
    fn new() {
        let context = Context::new();
        context.set_allow_unregistered_dialects(true);
        Builder::new("foo", Location::unknown(&context)).build();
    }

    #[test]
    fn name() {
        let context = Context::new();
        context.set_allow_unregistered_dialects(true);

        assert_eq!(
            Builder::new("foo", Location::unknown(&context))
                .build()
                .name(),
            Identifier::new(&context, "foo")
        );
    }

    #[test]
    fn result_error() {
        let context = Context::new();
        context.set_allow_unregistered_dialects(true);
        assert_eq!(
            Builder::new("foo", Location::unknown(&context))
                .build()
                .result(0)
                .unwrap_err(),
            Error::OperationResultPosition("\"foo\"() : () -> ()\n".into(), 0)
        );
    }

    #[test]
    fn region_none() {
        let context = Context::new();
        context.set_allow_unregistered_dialects(true);
        assert!(Builder::new("foo", Location::unknown(&context),)
            .build()
            .region(0)
            .is_none());
    }

    #[test]
    fn clone() {
        let context = Context::new();
        context.set_allow_unregistered_dialects(true);
        let operation = Builder::new("foo", Location::unknown(&context)).build();

        let _ = operation.clone();
    }

    #[test]
    fn display() {
        let context = Context::new();
        context.set_allow_unregistered_dialects(true);

        assert_eq!(
            Builder::new("foo", Location::unknown(&context),)
                .build()
                .to_string(),
            "\"foo\"() : () -> ()\n"
        );
    }

    #[test]
    fn debug() {
        let context = Context::new();
        context.set_allow_unregistered_dialects(true);

        assert_eq!(
            format!(
                "{:?}",
                Builder::new("foo", Location::unknown(&context)).build()
            ),
            "Operation(\n\"foo\"() : () -> ()\n)"
        );
    }

    #[test]
    fn debug_print() {
        let context = Context::new();
        context.set_allow_unregistered_dialects(true);

        let op = Builder::new("foo", Location::new(&context, "file.ext", 1, 1)).build();
        let debug_print = op.debug_print();

        assert_eq!(
            debug_print,
            r#""foo"() : () -> () loc(#loc)
#loc = loc("file.ext":1:1)
"#
        );
    }
}
