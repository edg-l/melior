//! Dialect conversion passes.

use super::Pass;
use crate::mlir_sys::{
    mlirCreateConversionArithToLLVMConversionPass, mlirCreateConversionConvertAffineForToGPU,
    mlirCreateConversionConvertAffineToStandard, mlirCreateConversionConvertAsyncToLLVM,
    mlirCreateConversionConvertControlFlowToLLVM, mlirCreateConversionConvertControlFlowToSPIRV,
    mlirCreateConversionConvertFuncToLLVM, mlirCreateConversionConvertMathToLLVM,
    mlirCreateConversionConvertMathToLibm, mlirCreateConversionConvertMathToSPIRV,
    mlirCreateConversionGpuToLLVMConversionPass, mlirCreateConversionSCFToControlFlow,
    mlirCreateConversionConvertIndexToLLVMPass
};

/// Creates a pass to convert the `arith` dialect to the `llvm` dialect.
pub fn convert_arithmetic_to_llvm() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionArithToLLVMConversionPass)
}

/// Creates a pass to convert the `cf` dialect to the `llvm` dialect.
pub fn convert_cf_to_llvm() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertControlFlowToLLVM)
}

/// Creates a pass to convert the `scf` dialect to the `cf` dialect.
pub fn convert_scf_to_cf() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionSCFToControlFlow)
}

/// Creates a pass to convert the `func` dialect to the `llvm` dialect.
pub fn convert_func_to_llvm() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertFuncToLLVM)
}

/// Creates a pass to convert the `math` dialect to the `llvm` dialect.
pub fn convert_math_to_llvm() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertMathToLLVM)
}

/// Creates a pass to convert the builtin index to the `llvm` dialect.
pub fn convert_index_to_llvm() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertIndexToLLVMPass)
}

/// Creates a pass to convert the `cf` dialect to the `spirv` dialect.
pub fn convert_cf_to_spirv() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertControlFlowToSPIRV)
}

/// Creates a pass to convert the `math` dialect to the `spirv` dialect.
pub fn convert_math_to_spirv() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertMathToSPIRV)
}

/// Creates a pass to convert the `math` dialect to the `libm` dialect.
pub fn convert_math_to_libm() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertMathToLibm)
}

/// Creates a pass to convert the `affine for` dialect to the `gpu` dialect.
pub fn convert_affine_for_to_gpu() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertAffineForToGPU)
}

/// Creates a pass to convert the `affine` dialect to the `standard` dialect.
pub fn convert_affine_to_standard() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertAffineToStandard)
}

/// Creates a pass to convert the `async` dialect to the `llvm` dialect.
pub fn convert_async_to_llvm() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertAsyncToLLVM)
}

/// Creates a pass to convert the `gpu` dialect to the `llvm` dialect.
pub fn convert_gpu_to_llvm() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionGpuToLLVMConversionPass)
}

/// Creates a pass to convert the `affiner for` dialect to the `gpu` dialect.
pub fn convert_affiner_for_to_gpu() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertAffineForToGPU)
}
