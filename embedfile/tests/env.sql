.mode qbox
.header on
.echo on
.bail on

select sqlite_version();

select vec_version();

select lembed_version();

select lines_version();

select name from pragma_function_list order by name;

select name from pragma_module_list order by name;

select compile_options from pragma_compile_options order by 1;
