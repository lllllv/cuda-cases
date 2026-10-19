set pagination off
set confirm off
set breakpoint pending on
set startup-with-shell off
set disable-randomization off
set cuda ptx_cache off
set cuda value_extrapolation off
break shared_elimination.cu:13
run
python
import gdb,json
record = {}
for command in ['p/x $pc', 'info line *$pc', 'info address x', 'ptype x', 'p x', 'p x.lo', 'p x.hi', 'p &x', 'p &x.lo', 'p &x.hi', 'info locals', 'show cuda ptx_cache', 'show cuda value_extrapolation']:
    try: output = gdb.execute(command, to_string=True)
    except gdb.error as exc: output = 'ERROR: ' + str(exc)
    record[command] = output
    print('COMMAND: ' + command + '\n' + output)
try:
    value = gdb.parse_and_eval('x')
    record['x_value'] = {'text': str(value), 'optimized_out': value.is_optimized_out}
except gdb.error as exc: record['x_value'] = {'error': str(exc)}
print('RECORD: ' + json.dumps(record))
end
continue
quit
