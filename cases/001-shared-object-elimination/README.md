# 001：第一阶段描述内存位置，第二阶段删除 shared 对象

状态：已实测（下列工具组合），调试信息传递与消费方案待设计。CUDA 实验日期：2026-10-18；归档日期：2026-10-19。

第一阶段的 PTX 中，`x` 位于 8 字节 shared 对象 `_ZZ6kernelE1x`。经 ptxas 后，该存储对象消失，cubin 保留 `x` 的变量 DIE，但位置变成 `DW_OP_addr 0xffffffffffffffff`。cuda-gdb 仍按这个地址读 shared memory 并报错，没有显示为 `<optimized out>`。

本条归档已有实验的源码和日志摘录。没有编辑 PTX、注入 DWARF、使用内联 PTX 或修改编译器来制造变换。

## 最小源码与范围

完整源码见 [shared_elimination.cu](shared_elimination.cu)。默认编译的函数主体是：

```cpp
struct Pair {
  int lo;
  int hi;
};

extern "C" __global__ void kernel(int* out, int a, int index) {
  __shared__ Pair x;
  out[0] = a;
}
```

`x` 没有初始化，也没有被程序读取。本例验证的是**存储对象删除后调试描述如何处理**，不验证恢复一个已计算出的值。对象位于 shared 空间，不是线程私有 local 栈对象。

同一源码定义 `USE_X` 后，执行 `x.lo = a + 1; x.hi = a + 2;`，再把两字段写到输出，作为保留对象的对照组。驱动只启动一个线程，`a = 17`。

## 实测环境

| 项目 | 版本或设置 |
| --- | --- |
| 第一阶段 | Clang CUDA 22.1.0，提交 `f0353f734bc53768dcf29157510b2867c4b86b40` |
| Clang 使用的 CUDA 头文件 | CUDA 12.2 |
| 中间格式 | PTX 8.2；DWARF 2 |
| 第二阶段 | ptxas 13.4.92，`-arch=sm_89 -g -O0 -v` |
| 调试器 | cuda-gdb 13.4，基于 GDB 16.3 |
| 设备与驱动 | RTX 4090 / `sm_89`；驱动 617.42；WSL Linux |
| 调试器设置 | `cuda ptx_cache off`，`cuda value_extrapolation off` |

第一阶段同样使用 `-O0`。该选项没有阻止第二阶段省略未使用的 shared 存储；输出不能证明 ptxas 内部由哪个 pass 或分配步骤完成。

## 各阶段观察

| 检查项 | 未使用 `x` | 定义 `USE_X` |
| --- | --- | --- |
| PTX 存储对象 | `_ZZ6kernelE1x[8]` | `_ZZ6kernelE1x[8]` |
| PTX 中 `x` 的位置 | `DW_OP_addr _ZZ6kernelE1x` | 相同 |
| cubin 的 `x` DIE | 保留 | 保留 |
| cubin 的位置 | `DW_OP_addr 0xffffffffffffffff` | `DW_OP_addr 0` |
| `.nv.shared.kernel` | 不存在 | 大小 8 字节 |
| `_ZZ6kernelE1x` 符号 | 不存在 | 大小 8 字节 |
| Driver API 查询静态 shared 大小 | 0 | 8 |
| cuda-gdb：`p x` | 无效 shared 地址读取错误 | `{lo = 18, hi = 19}` |
| cuda-gdb：`p &x` | `(@shared Pair *) 0xffffffffffffffff` | `(@shared Pair *) 0x0` |
| kernel 输出校验 | `17, 0`，通过 | `18, 19`，通过 |

第二个输出元素在启动前由驱动清零，因此未使用组的 `out[1] = 0` 不是对 `x` 的任何观察。

### PTX：符号化的内存位置

以下是编译器输出的相关片段；注释只用于本文解释，不是需要写回 PTX 的内容：

```text
.shared .align 4 .b8 _ZZ6kernelE1x[8];

// x 的 DIE：DW_AT_address_class = 8
// DW_AT_location
.b8 9                    // 表达式长度
.b8 3                    // DW_OP_addr
.b64 _ZZ6kernelE1x
```

这里的 A 就是 `_ZZ6kernelE1x`。解析未完成的符号时若暂填 0，必须同时看符号修正记录，不能把这个 0 当作真实地址。

### cubin：对象删除，但位置仍是非空的地址表达式

```text
DW_AT_name          : x
DW_AT_address_class : 8
DW_AT_location      : 9 byte block: 3 ff ff ff ff ff ff ff ff
                      (DW_OP_addr: ffffffffffffffff)
```

这不是空位置表达式，也没有变成寄存器或 `DW_OP_piece` 描述。对象删除由 section、符号表和运行时 shared 大小交叉确认。`.rela.debug_info` 仅保留函数地址重定位，没有 `x` 的位置重定位。产物仍有用于参数等的 16 字节 local 栈，与这个 shared 对象不同。

### cuda-gdb：错误还会影响成员表达式

未使用组在源码第 16 行 `out[0] = a` 停下，线程 `(0,0,0)`：

```text
(cuda-gdb) info address x
Symbol "x" is static storage at address 0xffffffffffffffff.

(cuda-gdb) p x
Error: read_shared_memory(0, 0, 0): failed to read shared memory at
address 0xffffffffffffffff size 8, error=CUDBG_ERROR_INVALID_MEMORY_ACCESS

(cuda-gdb) p &x
(@shared Pair *) 0xffffffffffffffff
```

`p x.lo` 同样读取失败；`p &x.hi` 得到 `0x3`；`p x.hi` 在该次运行输出 `0`。地址结果与全 1 基地址加 4 字节成员偏移后回绕相符，这是对观察的解释。该数字不能当作已删除变量的有效值，也不应作为其他运行的预期结果。

对照组在第 13 行、完成两个字段赋值后停下，能读到 `{lo = 18, hi = 19}`。其地址 0 是 **shared 空间的有效偏移**。

## 复现

从本案例目录，在 Bash 中执行。需要上述版本的 Clang、CUDA 12.2 头文件、ptxas/cuda-gdb，以及支持 `sm_89` 的 GPU 和可用驱动。更换版本或架构后，应重新记录观察。

```bash
cd cases/001-shared-object-elimination
clang_cuda="${CLANG_CUDA:-clang++}"
cuda_headers="${CUDA_HEADERS:-/usr/local/cuda-12.2}"
ptxas_bin="${PTXAS:-/usr/local/cuda-13.4/bin/ptxas}"
cuda_gdb="${CUDA_GDB:-/usr/local/cuda-13.4/bin/cuda-gdb}"
mkdir -p build/unused build/used

for mode in unused used; do
  defines=()
  if [[ "$mode" == used ]]; then defines=(-DUSE_X); fi
  "$clang_cuda" -x cuda --cuda-device-only \
    --cuda-path="$cuda_headers" --cuda-gpu-arch=sm_89 \
    -S -g -gdwarf-2 -O0 "${defines[@]}" \
    shared_elimination.cu -o "build/$mode/kernel.ptx" || exit 1
  sha256sum "build/$mode/kernel.ptx" > "build/$mode/ptx.sha256"
  "$ptxas_bin" -arch=sm_89 -g -O0 -v \
    "build/$mode/kernel.ptx" -o "build/$mode/kernel.cubin" || exit 1
  sha256sum -c "build/$mode/ptx.sha256" || exit 1
  readelf --debug-dump=info "build/$mode/kernel.cubin" > "build/$mode/dwarf.txt"
  readelf -SW "build/$mode/kernel.cubin" > "build/$mode/sections.txt"
  readelf -sW "build/$mode/kernel.cubin" > "build/$mode/symbols.txt"
  readelf -rW "build/$mode/kernel.cubin" > "build/$mode/relocations.txt"
done
```

用 [driver.cpp](driver.cpp) 查询存储并验证输出。下面的库路径是原 WSL 环境的路径；其他 Linux 环境将 `CUDA_DRIVER_LIB_DIR` 指向真实 `libcuda.so` 所在目录。驱动程序用 CUDA 12.2 头文件编译。

```bash
driver_lib_dir="${CUDA_DRIVER_LIB_DIR:-/usr/lib/wsl/lib}"
g++ -std=c++17 -g -O0 driver.cpp -I"$cuda_headers/include" \
  -L"$driver_lib_dir" -Wl,-rpath,"$driver_lib_dir" -lcuda -o build/driver
./build/driver build/unused/kernel.cubin 17 17 0
./build/driver build/used/kernel.cubin 17 18 19

"$cuda_gdb" -nx -q -batch -x unused.gdb \
  --args ./build/driver build/unused/kernel.cubin 17 17 0
"$cuda_gdb" -nx -q -batch -x used.gdb \
  --args ./build/driver build/used/kernel.cubin 17 18 19
sha256sum -c build/unused/ptx.sha256 build/used/ptx.sha256
```

[unused.gdb](unused.gdb) 和 [used.gdb](used.gdb) 会捕获单条求值命令的错误后继续观察，并继续运行 kernel。断点行号对应仓库中未经修改的源码。

## 归档证据

[evidence.json](evidence.json) 保存既有两组实验的源码/PTX SHA-256、PTX 符号引用、readelf 的 section/符号/重定位输出、变量 DIE 摘录、运行时输出，以及逐命令的 cuda-gdb 结果。调试器摘要保留原始命令输出，不包括机器绝对源码路径和线程启动日志。

原实验在生成 PTX 后没有写入该文件，处理前后 SHA-256 相同。归档中的 PTX 哈希用于标识原实验；重编译会因路径等调试信息不同而得到不同哈希。应比较本次运行前后是否相同，而不是要求与归档哈希相同。本仓库保存源码和文本证据，不保存完整 PTX/cubin 二进制产物。

## 设计时需要特别处理

1. **存储对象失效。** 第一阶段的 `x → A` 不能在第二阶段删除 A 后继续作为可用内存位置。变量名字、类型和作用域仍可保留。
2. **删除标记的生产与消费约定。** 若保留特殊地址作为标记，必须明确宽度、适用地址空间和消费者行为；必须在读内存及成员偏移运算之前识别失效。
3. **GPU 地址空间。** shared 偏移 0 在本对照组中有效，不能把 CPU 上针对空地址的规则直接套过来。
4. **值与地址分别处理。** 存储消失后，值可能不可用，也可能仍是常量、寄存器值或多个片段。能够显示值不代表仍有可取的内存地址。
5. **回归验证覆盖成员访问。** 同时检查 `x`、`&x`、`x.lo`、`x.hi` 和成员地址；避免只修复整对象显示，却让成员表达式继续使用失效地址。

对于变量定义 DIE，无法描述的对象可以使用空位置表达式，或在没有常量值等描述时省略位置；已知常量可用 `DW_AT_const_value`，分散位置可用寄存器描述和 `DW_OP_piece`。位置仅在部分 PC 有效时，应限制其有效范围。参见 [DWARF 5 §2.6、§4.1](https://dwarfstd.org/doc/DWARF5.pdf)。这些是可用的表达机制，不是本例中 ptxas 已执行的处理。

[DWARF 提案 200609.1](https://dwarfstd.org/issues/200609.1.html) 已被接受用于 DWARF 6，规定最大可表示地址作为不存在实体的标记。这提供了 tombstone 的协议背景；不能据此证明该 DWARF 2 样本中 ptxas 的意图或 cuda-gdb 已支持对应约定。

## CPU 对照与结论边界

2026-10-19 的既有 x86-64 CPU 对照中，GCC 11.4.0 / Clang 22.1.0 的 `-O2 -g -gdwarf-4` 构建对未使用变量保留 DIE、去掉位置，GDB 12.1 显示 `<optimized out>`；常量和分散在寄存器的结构体则分别出现 `DW_AT_const_value` 和 `DW_OP_piece`，能显示值但不能取得内存地址。

另一个更接近“后续阶段删除”的 CPU 对照使用同一份 GCC `-O0 -fdata-sections` 对象文件：GNU ld 2.38 的 `--gc-sections` 删除未使用 static 对象后保留 `DW_OP_addr 0`；GDB 12.1 跳过该变量，查询时报 `No symbol "x" in current context`。可对照其 [new_symbol() 源码](https://gnu.googlesource.com/binutils-gdb/+/refs/tags/gdb-12.1-release/gdb/dwarf2/read.c)，检查静态变量地址为 0 且没有对应 section 的分支。这是补充背景；本条附带的复现源码针对 CUDA。

- 实测结论限于本工具组合和未使用的 shared 对象；不能推广为所有内存对象删除的统一行为。
- 调试器读无效地址、甚至从成员表达式得到数字，都不算成功恢复变量。该端到端表现需要处理，但现有证据不能把责任单独归给 Clang、ptxas 或 cuda-gdb。
- **尚未找到第一阶段仍描述内存、第二阶段将该对象拆成寄存器片段的自然案例。** Clang 在 PTX 中已输出 `DW_OP_piece` 的例子属于第一阶段拆分，不能算本问题的验证。
- ptxas 内部调试对象、算法和 pass 未知；全 1 地址不能直接称为 DWARF 2 通用的“optimized out”编码。
