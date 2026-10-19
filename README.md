# CUDA 调试信息案例

记录调试信息传递中需要特别考虑的 case。

| 编号 | Case | CUDA 源文件 |
| --- | --- | --- |
| 001 | 第一阶段描述的内存对象，被第二阶段删除后，变量位置如何处理 | [shared_elimination.cu](cases/001-shared-object-elimination/shared_elimination.cu) |

## 001：内存对象被第二阶段删除

**场景：** kernel 声明 `__shared__ Pair x`，但不使用它。Clang 生成的 PTX 中仍有 8 字节对象 `_ZZ6kernelE1x`，`x` 的 DWARF 位置为 `DW_OP_addr _ZZ6kernelE1x`。

**实际观察：**

- ptxas 生成 cubin 后，对象的 section 和符号消失，运行时查询 shared 大小为 0。
- `x` 的变量 DIE 仍保留，但位置变成 `DW_OP_addr 0xffffffffffffffff`。
- cuda-gdb 按该地址读取 shared memory，报 `CUDBG_ERROR_INVALID_MEMORY_ACCESS`，没有显示为 `<optimized out>`。
- `p &x.hi` 得到 `0x3`，与无效基地址加成员偏移后回绕相符；不能把随后读出的数字当作变量的有效值。

**对照：** 同一源文件定义 `USE_X`，初始化并使用 `x` 后，cubin 保留 8 字节 shared 存储。`x` 的位置为 shared 偏移 0；输入 `a = 17` 时，cuda-gdb 正常显示 `{lo = 18, hi = 19}`。

**需要考虑：** 第二阶段删除存储后，应更新或使原变量位置失效；若采用特殊地址作为删除标记，调试器必须在读内存、计算成员地址前识别它。shared 偏移 0 可以有效，不能直接套用 CPU 的空地址判断。

**实验条件：** Clang 22.1.0（CUDA 12.2 头文件，`-g -gdwarf-2 -O0`）→ ptxas 13.4.92（`-arch=sm_89 -g -O0`）；cuda-gdb 13.4；RTX 4090。PTX 未经人工修改。实验日期：2026-10-18。

**范围：** 这里只验证了未使用 shared 对象被删除。`x` 未初始化、未读取，不涉及恢复已计算出的值；尚未验证第二阶段将内存对象拆成寄存器片段。不能把本例推广为所有 CUDA 工具链的统一行为。
