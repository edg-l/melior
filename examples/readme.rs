use melior_next::{dialect, ir::*, pass, utility::*, Context, ExecutionEngine};

fn main() {
    let registry = dialect::Registry::new();
    register_all_dialects(&registry);

    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.get_or_load_dialect("func");
    register_all_llvm_translations(&context);

    let location = Location::unknown(&context);
    let mut module = Module::new(location);

    let integer_type = Type::integer(&context, 64);

    let function = {
        let mut region = Region::new();
        let mut block = Block::new(&[(integer_type, location), (integer_type, location)]);

        let sum = block.append_operation(
            operation::Builder::new("arith.addi", location)
                .add_operands(&[
                    block.argument(0).unwrap().into(),
                    block.argument(1).unwrap().into(),
                ])
                .add_results(&[integer_type])
                .build(),
        );

        block.append_operation(
            operation::Builder::new("func.return", Location::unknown(&context))
                .add_operands(&[sum.borrow().result(0).unwrap().into()])
                .build(),
        );

        region.append_block(block);

        operation::Builder::new("func.func", Location::unknown(&context))
            .add_attributes(
                &NamedAttribute::new_parsed_vec(
                    &context,
                    &[
                        ("function_type", "(i64, i64) -> i64"),
                        ("sym_name", "\"add\""),
                        ("llvm.emit_c_interface", "unit"),
                    ],
                )
                .unwrap(),
            )
            .add_regions(vec![region])
            .build()
    };

    module.body.append_operation(function);

    assert!(module.operation.verify());

    let pass_manager = pass::Manager::new(&context);
    register_all_passes();
    pass_manager.add_pass(pass::conversion::convert_scf_to_cf());
    pass_manager.add_pass(pass::conversion::convert_cf_to_llvm());
    pass_manager.add_pass(pass::conversion::convert_func_to_llvm());
    pass_manager.add_pass(pass::conversion::convert_arithmetic_to_llvm());
    pass_manager.enable_verifier(true);
    pass_manager.run(&mut module).unwrap();

    let engine = ExecutionEngine::new(&module, 2, &[], false);

    let mut argument1: i64 = 2;
    let mut argument2: i64 = 4;
    let mut result: i64 = -1;

    unsafe {
        engine
            .invoke_packed(
                "add",
                &mut [
                    &mut argument1 as *mut i64 as *mut (),
                    &mut argument2 as *mut i64 as *mut (),
                    &mut result as *mut i64 as *mut (),
                ],
            )
            .unwrap();
    };

    assert_eq!(result, 6);
}
