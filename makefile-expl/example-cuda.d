example-cuda.cuo : example-cuda.cu \
    /usr/include/stdc-predef.h \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/cuda_runtime.h \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/crt/host_config.h \
    /usr/include/features.h \
    /usr/include/features-time64.h \
    /usr/include/x86_64-linux-gnu/bits/wordsize.h \
    /usr/include/x86_64-linux-gnu/bits/timesize.h \
    /usr/include/x86_64-linux-gnu/sys/cdefs.h \
    /usr/include/x86_64-linux-gnu/bits/long-double.h \
    /usr/include/x86_64-linux-gnu/gnu/stubs.h \
    /usr/include/x86_64-linux-gnu/gnu/stubs-64.h \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/builtin_types.h \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/device_types.h \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/crt/host_defines.h \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/driver_types.h \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/vector_types.h \
    /usr/lib/gcc/x86_64-linux-gnu/11/include/limits.h \
    /usr/lib/gcc/x86_64-linux-gnu/11/include/syslimits.h \
    /usr/include/limits.h \
    /usr/include/x86_64-linux-gnu/bits/libc-header-start.h \
    /usr/include/x86_64-linux-gnu/bits/posix1_lim.h \
    /usr/include/x86_64-linux-gnu/bits/local_lim.h \
    /usr/include/linux/limits.h \
    /usr/include/x86_64-linux-gnu/bits/pthread_stack_min-dynamic.h \
    /usr/include/x86_64-linux-gnu/bits/posix2_lim.h \
    /usr/include/x86_64-linux-gnu/bits/xopen_lim.h \
    /usr/include/x86_64-linux-gnu/bits/uio_lim.h \
    /usr/lib/gcc/x86_64-linux-gnu/11/include/stddef.h \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/surface_types.h \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/texture_types.h \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/library_types.h \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/channel_descriptor.h \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/cuda_runtime_api.h \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/cuda_device_runtime_api.h \
    /usr/include/c++/11/stdlib.h \
    /usr/include/c++/11/cstdlib \
    /usr/include/x86_64-linux-gnu/c++/11/bits/c++config.h \
    /usr/include/x86_64-linux-gnu/c++/11/bits/os_defines.h \
    /usr/include/x86_64-linux-gnu/c++/11/bits/cpu_defines.h \
    /usr/include/c++/11/pstl/pstl_config.h \
    /usr/include/stdlib.h \
    /usr/include/x86_64-linux-gnu/bits/waitflags.h \
    /usr/include/x86_64-linux-gnu/bits/waitstatus.h \
    /usr/include/x86_64-linux-gnu/bits/floatn.h \
    /usr/include/x86_64-linux-gnu/bits/floatn-common.h \
    /usr/include/x86_64-linux-gnu/bits/types/locale_t.h \
    /usr/include/x86_64-linux-gnu/bits/types/__locale_t.h \
    /usr/include/x86_64-linux-gnu/sys/types.h \
    /usr/include/x86_64-linux-gnu/bits/types.h \
    /usr/include/x86_64-linux-gnu/bits/typesizes.h \
    /usr/include/x86_64-linux-gnu/bits/time64.h \
    /usr/include/x86_64-linux-gnu/bits/types/clock_t.h \
    /usr/include/x86_64-linux-gnu/bits/types/clockid_t.h \
    /usr/include/x86_64-linux-gnu/bits/types/time_t.h \
    /usr/include/x86_64-linux-gnu/bits/types/timer_t.h \
    /usr/include/x86_64-linux-gnu/bits/stdint-intn.h \
    /usr/include/endian.h \
    /usr/include/x86_64-linux-gnu/bits/endian.h \
    /usr/include/x86_64-linux-gnu/bits/endianness.h \
    /usr/include/x86_64-linux-gnu/bits/byteswap.h \
    /usr/include/x86_64-linux-gnu/bits/uintn-identity.h \
    /usr/include/x86_64-linux-gnu/sys/select.h \
    /usr/include/x86_64-linux-gnu/bits/select.h \
    /usr/include/x86_64-linux-gnu/bits/types/sigset_t.h \
    /usr/include/x86_64-linux-gnu/bits/types/__sigset_t.h \
    /usr/include/x86_64-linux-gnu/bits/types/struct_timeval.h \
    /usr/include/x86_64-linux-gnu/bits/types/struct_timespec.h \
    /usr/include/x86_64-linux-gnu/bits/pthreadtypes.h \
    /usr/include/x86_64-linux-gnu/bits/thread-shared-types.h \
    /usr/include/x86_64-linux-gnu/bits/pthreadtypes-arch.h \
    /usr/include/x86_64-linux-gnu/bits/atomic_wide_counter.h \
    /usr/include/x86_64-linux-gnu/bits/struct_mutex.h \
    /usr/include/x86_64-linux-gnu/bits/struct_rwlock.h \
    /usr/include/alloca.h \
    /usr/include/x86_64-linux-gnu/bits/stdlib-float.h \
    /usr/include/c++/11/bits/std_abs.h \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/driver_functions.h \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/vector_functions.h \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/vector_functions.hpp \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/crt/common_functions.h \
    /usr/include/string.h \
    /usr/include/strings.h \
    /usr/include/time.h \
    /usr/include/x86_64-linux-gnu/bits/time.h \
    /usr/include/x86_64-linux-gnu/bits/timex.h \
    /usr/include/x86_64-linux-gnu/bits/types/struct_tm.h \
    /usr/include/x86_64-linux-gnu/bits/types/struct_itimerspec.h \
    /usr/include/c++/11/new \
    /usr/include/c++/11/bits/exception.h \
    /usr/include/stdio.h \
    /usr/lib/gcc/x86_64-linux-gnu/11/include/stdarg.h \
    /usr/include/x86_64-linux-gnu/bits/types/__fpos_t.h \
    /usr/include/x86_64-linux-gnu/bits/types/__mbstate_t.h \
    /usr/include/x86_64-linux-gnu/bits/types/__fpos64_t.h \
    /usr/include/x86_64-linux-gnu/bits/types/__FILE.h \
    /usr/include/x86_64-linux-gnu/bits/types/FILE.h \
    /usr/include/x86_64-linux-gnu/bits/types/struct_FILE.h \
    /usr/include/x86_64-linux-gnu/bits/types/cookie_io_functions_t.h \
    /usr/include/x86_64-linux-gnu/bits/stdio_lim.h \
    /usr/include/assert.h \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/crt/math_functions.h \
    /usr/include/c++/11/math.h \
    /usr/include/c++/11/cmath \
    /usr/include/c++/11/bits/cpp_type_traits.h \
    /usr/include/c++/11/ext/type_traits.h \
    /usr/include/math.h \
    /usr/include/x86_64-linux-gnu/bits/math-vector.h \
    /usr/include/x86_64-linux-gnu/bits/libm-simd-decl-stubs.h \
    /usr/include/x86_64-linux-gnu/bits/flt-eval-method.h \
    /usr/include/x86_64-linux-gnu/bits/fp-logb.h \
    /usr/include/x86_64-linux-gnu/bits/fp-fast.h \
    /usr/include/x86_64-linux-gnu/bits/mathcalls-helper-functions.h \
    /usr/include/x86_64-linux-gnu/bits/mathcalls.h \
    /usr/include/x86_64-linux-gnu/bits/mathcalls-narrow.h \
    /usr/include/x86_64-linux-gnu/bits/iscanonical.h \
    /usr/include/c++/11/bits/specfun.h \
    /usr/include/c++/11/bits/stl_algobase.h \
    /usr/include/c++/11/bits/functexcept.h \
    /usr/include/c++/11/bits/exception_defines.h \
    /usr/include/c++/11/ext/numeric_traits.h \
    /usr/include/c++/11/bits/stl_pair.h \
    /usr/include/c++/11/bits/move.h \
    /usr/include/c++/11/type_traits \
    /usr/include/c++/11/bits/stl_iterator_base_types.h \
    /usr/include/c++/11/bits/stl_iterator_base_funcs.h \
    /usr/include/c++/11/bits/concept_check.h \
    /usr/include/c++/11/debug/assertions.h \
    /usr/include/c++/11/bits/stl_iterator.h \
    /usr/include/c++/11/bits/ptr_traits.h \
    /usr/include/c++/11/debug/debug.h \
    /usr/include/c++/11/bits/predefined_ops.h \
    /usr/include/c++/11/limits \
    /usr/include/c++/11/tr1/gamma.tcc \
    /usr/include/c++/11/tr1/special_function_util.h \
    /usr/include/c++/11/tr1/bessel_function.tcc \
    /usr/include/c++/11/tr1/beta_function.tcc \
    /usr/include/c++/11/tr1/ell_integral.tcc \
    /usr/include/c++/11/tr1/exp_integral.tcc \
    /usr/include/c++/11/tr1/hypergeometric.tcc \
    /usr/include/c++/11/tr1/legendre_function.tcc \
    /usr/include/c++/11/tr1/modified_bessel_func.tcc \
    /usr/include/c++/11/tr1/poly_hermite.tcc \
    /usr/include/c++/11/tr1/poly_laguerre.tcc \
    /usr/include/c++/11/tr1/riemann_zeta.tcc \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/crt/math_functions.hpp \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/crt/device_functions.h \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/crt/device_functions.hpp \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/device_atomic_functions.h \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/device_atomic_functions.hpp \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/crt/device_double_functions.h \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/sm_20_atomic_functions.h \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/sm_20_atomic_functions.hpp \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.h \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.hpp \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/sm_35_atomic_functions.h \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.hpp \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/sm_20_intrinsics.hpp \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/sm_30_intrinsics.hpp \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/sm_32_intrinsics.hpp \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/sm_35_intrinsics.h \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/sm_61_intrinsics.hpp \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/crt/sm_70_rt.hpp \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/crt/sm_80_rt.h \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/crt/sm_80_rt.hpp \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/crt/sm_90_rt.h \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/crt/sm_90_rt.hpp \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/texture_indirect_functions.h \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/surface_indirect_functions.h \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/crt/cudacc_ext.h \
    /usr/local/cuda/bin/../targets/x86_64-linux/include/device_launch_parameters.h \
    /usr/include/c++/11/functional \
    /usr/include/c++/11/bits/stl_function.h \
    /usr/include/c++/11/backward/binders.h \
    /usr/include/c++/11/tuple \
    /usr/include/c++/11/utility \
    /usr/include/c++/11/bits/stl_relops.h \
    /usr/include/c++/11/initializer_list \
    /usr/include/c++/11/array \
    /usr/include/c++/11/bits/range_access.h \
    /usr/include/c++/11/bits/uses_allocator.h \
    /usr/include/c++/11/bits/invoke.h \
    /usr/include/c++/11/bits/functional_hash.h \
    /usr/include/c++/11/bits/hash_bytes.h \
    /usr/include/c++/11/bits/refwrap.h \
    /usr/include/c++/11/bits/std_function.h \
    /usr/include/c++/11/typeinfo \
    /usr/include/c++/11/unordered_map \
    /usr/include/c++/11/bits/allocator.h \
    /usr/include/x86_64-linux-gnu/c++/11/bits/c++allocator.h \
    /usr/include/c++/11/ext/new_allocator.h \
    /usr/include/c++/11/bits/memoryfwd.h \
    /usr/include/c++/11/ext/alloc_traits.h \
    /usr/include/c++/11/bits/alloc_traits.h \
    /usr/include/c++/11/bits/stl_construct.h \
    /usr/include/c++/11/ext/aligned_buffer.h \
    /usr/include/c++/11/bits/hashtable.h \
    /usr/include/c++/11/bits/hashtable_policy.h \
    /usr/include/c++/11/bits/enable_special_members.h \
    /usr/include/c++/11/bits/node_handle.h \
    /usr/include/c++/11/bits/unordered_map.h \
    /usr/include/c++/11/bits/erase_if.h \
    /usr/include/c++/11/vector \
    /usr/include/c++/11/bits/stl_uninitialized.h \
    /usr/include/c++/11/bits/stl_vector.h \
    /usr/include/c++/11/bits/stl_bvector.h \
    /usr/include/c++/11/bits/vector.tcc \
    /usr/include/c++/11/bits/stl_algo.h \
    /usr/include/c++/11/bits/algorithmfwd.h \
    /usr/include/c++/11/bits/stl_heap.h \
    /usr/include/c++/11/bits/stl_tempbuf.h \
    /usr/include/c++/11/bits/uniform_int_dist.h \
    /usr/include/c++/11/iostream \
    /usr/include/c++/11/ostream \
    /usr/include/c++/11/ios \
    /usr/include/c++/11/iosfwd \
    /usr/include/c++/11/bits/stringfwd.h \
    /usr/include/c++/11/bits/postypes.h \
    /usr/include/c++/11/cwchar \
    /usr/include/wchar.h \
    /usr/include/x86_64-linux-gnu/bits/wchar.h \
    /usr/include/x86_64-linux-gnu/bits/types/wint_t.h \
    /usr/include/x86_64-linux-gnu/bits/types/mbstate_t.h \
    /usr/include/c++/11/exception \
    /usr/include/c++/11/bits/exception_ptr.h \
    /usr/include/c++/11/bits/cxxabi_init_exception.h \
    /usr/include/c++/11/bits/nested_exception.h \
    /usr/include/c++/11/bits/char_traits.h \
    /usr/include/c++/11/cstdint \
    /usr/lib/gcc/x86_64-linux-gnu/11/include/stdint.h \
    /usr/include/stdint.h \
    /usr/include/x86_64-linux-gnu/bits/stdint-uintn.h \
    /usr/include/c++/11/bits/localefwd.h \
    /usr/include/x86_64-linux-gnu/c++/11/bits/c++locale.h \
    /usr/include/c++/11/clocale \
    /usr/include/locale.h \
    /usr/include/x86_64-linux-gnu/bits/locale.h \
    /usr/include/c++/11/cctype \
    /usr/include/ctype.h \
    /usr/include/c++/11/bits/ios_base.h \
    /usr/include/c++/11/ext/atomicity.h \
    /usr/include/x86_64-linux-gnu/c++/11/bits/gthr.h \
    /usr/include/x86_64-linux-gnu/c++/11/bits/gthr-default.h \
    /usr/include/pthread.h \
    /usr/include/sched.h \
    /usr/include/x86_64-linux-gnu/bits/sched.h \
    /usr/include/x86_64-linux-gnu/bits/types/struct_sched_param.h \
    /usr/include/x86_64-linux-gnu/bits/cpu-set.h \
    /usr/include/x86_64-linux-gnu/bits/setjmp.h \
    /usr/include/x86_64-linux-gnu/bits/types/struct___jmp_buf_tag.h \
    /usr/include/x86_64-linux-gnu/c++/11/bits/atomic_word.h \
    /usr/include/x86_64-linux-gnu/sys/single_threaded.h \
    /usr/include/c++/11/bits/locale_classes.h \
    /usr/include/c++/11/string \
    /usr/include/c++/11/bits/ostream_insert.h \
    /usr/include/c++/11/bits/cxxabi_forced.h \
    /usr/include/c++/11/bits/basic_string.h \
    /usr/include/c++/11/string_view \
    /usr/include/c++/11/bits/string_view.tcc \
    /usr/include/c++/11/ext/string_conversions.h \
    /usr/include/c++/11/cstdio \
    /usr/include/c++/11/cerrno \
    /usr/include/errno.h \
    /usr/include/x86_64-linux-gnu/bits/errno.h \
    /usr/include/linux/errno.h \
    /usr/include/x86_64-linux-gnu/asm/errno.h \
    /usr/include/asm-generic/errno.h \
    /usr/include/asm-generic/errno-base.h \
    /usr/include/x86_64-linux-gnu/bits/types/error_t.h \
    /usr/include/c++/11/bits/charconv.h \
    /usr/include/c++/11/bits/basic_string.tcc \
    /usr/include/c++/11/bits/locale_classes.tcc \
    /usr/include/c++/11/system_error \
    /usr/include/x86_64-linux-gnu/c++/11/bits/error_constants.h \
    /usr/include/c++/11/stdexcept \
    /usr/include/c++/11/streambuf \
    /usr/include/c++/11/bits/streambuf.tcc \
    /usr/include/c++/11/bits/basic_ios.h \
    /usr/include/c++/11/bits/locale_facets.h \
    /usr/include/c++/11/cwctype \
    /usr/include/wctype.h \
    /usr/include/x86_64-linux-gnu/bits/wctype-wchar.h \
    /usr/include/x86_64-linux-gnu/c++/11/bits/ctype_base.h \
    /usr/include/c++/11/bits/streambuf_iterator.h \
    /usr/include/x86_64-linux-gnu/c++/11/bits/ctype_inline.h \
    /usr/include/c++/11/bits/locale_facets.tcc \
    /usr/include/c++/11/bits/basic_ios.tcc \
    /usr/include/c++/11/bits/ostream.tcc \
    /usr/include/c++/11/istream \
    /usr/include/c++/11/bits/istream.tcc \
    /home/lali/.local/include/TNL/Containers/Array.h \
    /usr/include/c++/11/list \
    /usr/include/c++/11/bits/stl_list.h \
    /usr/include/c++/11/bits/allocated_ptr.h \
    /usr/include/c++/11/bits/list.tcc \
    /home/lali/.local/include/TNL/File.h \
    /usr/include/c++/11/fstream \
    /usr/include/c++/11/bits/codecvt.h \
    /usr/include/x86_64-linux-gnu/c++/11/bits/basic_file.h \
    /usr/include/x86_64-linux-gnu/c++/11/bits/c++io.h \
    /usr/include/c++/11/bits/fstream.tcc \
    /home/lali/.local/include/TNL/String.h \
    /usr/include/c++/11/sstream \
    /usr/include/c++/11/bits/sstream.tcc \
    /home/lali/.local/include/TNL/String.hpp \
    /home/lali/.local/include/TNL/Assert.h \
    /home/lali/.local/include/TNL/Cuda/CudaCallable.h \
    /home/lali/.local/include/TNL/Math.h \
    /usr/include/c++/11/algorithm \
    /usr/include/c++/11/pstl/glue_algorithm_defs.h \
    /usr/include/c++/11/pstl/execution_defs.h \
    /home/lali/.local/include/TNL/TypeTraits.h \
    /home/lali/.local/include/TNL/Allocators/Host.h \
    /usr/include/c++/11/memory \
    /usr/include/c++/11/bits/stl_raw_storage_iter.h \
    /usr/include/c++/11/bits/align.h \
    /usr/include/c++/11/bit \
    /usr/include/c++/11/bits/unique_ptr.h \
    /usr/include/c++/11/bits/shared_ptr.h \
    /usr/include/c++/11/bits/shared_ptr_base.h \
    /usr/include/c++/11/ext/concurrence.h \
    /usr/include/c++/11/bits/shared_ptr_atomic.h \
    /usr/include/c++/11/bits/atomic_base.h \
    /usr/include/c++/11/bits/atomic_lockfree_defines.h \
    /usr/include/c++/11/backward/auto_ptr.h \
    /usr/include/c++/11/pstl/glue_memory_defs.h \
    /home/lali/.local/include/TNL/Allocators/Cuda.h \
    /home/lali/.local/include/TNL/Exceptions/CudaBadAlloc.h \
    /home/lali/.local/include/TNL/Exceptions/CudaSupportMissing.h \
    /home/lali/.local/include/TNL/Cuda/CheckDevice.h \
    /home/lali/.local/include/TNL/Exceptions/CudaRuntimeError.h \
    /home/lali/.local/include/TNL/File.hpp \
    /home/lali/.local/include/TNL/Cuda/LaunchHelpers.h \
    /home/lali/.local/include/TNL/DiscreteMath.h \
    /home/lali/.local/include/TNL/Cuda/DummyDefs.h \
    /home/lali/.local/include/TNL/Exceptions/FileSerializationError.h \
    /home/lali/.local/include/TNL/Exceptions/FileDeserializationError.h \
    /home/lali/.local/include/TNL/Exceptions/NotImplementedError.h \
    /home/lali/.local/include/TNL/Allocators/Default.h \
    /home/lali/.local/include/TNL/Devices/Sequential.h \
    /home/lali/.local/include/TNL/Devices/Host.h \
    /home/lali/.local/include/TNL/Config/ConfigDescription.h \
    /home/lali/.local/include/TNL/Config/ConfigEntry.h \
    /usr/include/c++/11/optional \
    /home/lali/.local/include/TNL/Config/ConfigEntryBase.h \
    /home/lali/.local/include/TNL/Config/ConfigEntryType.h \
    /usr/include/c++/11/variant \
    /usr/include/c++/11/bits/parse_numbers.h \
    /home/lali/.local/include/TNL/Config/ConfigEntryList.h \
    /home/lali/.local/include/TNL/Config/ConfigDelimiter.h \
    /home/lali/.local/include/TNL/Exceptions/ConfigError.h \
    /home/lali/.local/include/TNL/Config/ParameterContainer.h \
    /home/lali/.local/include/TNL/TypeInfo.h \
    /usr/include/c++/11/cxxabi.h \
    /usr/include/x86_64-linux-gnu/c++/11/bits/cxxabi_tweaks.h \
    /usr/lib/gcc/x86_64-linux-gnu/11/include/omp.h \
    /home/lali/.local/include/TNL/Devices/Cuda.h \
    /home/lali/.local/include/TNL/Cuda/KernelLaunch.h \
    /home/lali/.local/include/TNL/Containers/ArrayView.h \
    /home/lali/.local/include/TNL/Containers/ArrayView.hpp \
    /home/lali/.local/include/TNL/Algorithms/copy.h \
    /home/lali/.local/include/TNL/Algorithms/detail/Copy.h \
    /home/lali/.local/include/TNL/Algorithms/detail/Copy.hpp \
    /home/lali/.local/include/TNL/Algorithms/parallelFor.h \
    /home/lali/.local/include/TNL/Algorithms/detail/ParallelFor1D.h \
    /home/lali/.local/include/TNL/Cuda/DeviceInfo.h \
    /home/lali/.local/include/TNL/Cuda/DeviceInfo.hpp \
    /home/lali/.local/include/TNL/Algorithms/detail/ParallelFor2D.h \
    /home/lali/.local/include/TNL/Algorithms/detail/ParallelFor3D.h \
    /home/lali/.local/include/TNL/Algorithms/equal.h \
    /home/lali/.local/include/TNL/Algorithms/detail/Equal.h \
    /home/lali/.local/include/TNL/Algorithms/detail/Equal.hpp \
    /home/lali/.local/include/TNL/Algorithms/reduce.h \
    /home/lali/.local/include/TNL/Functional.h \
    /home/lali/.local/include/TNL/Algorithms/detail/Reduction.h \
    /home/lali/.local/include/TNL/Algorithms/detail/Reduction.hpp \
    /home/lali/.local/include/TNL/Algorithms/detail/CudaReductionKernel.h \
    /home/lali/.local/include/TNL/Algorithms/CudaReductionBuffer.h \
    /home/lali/.local/include/TNL/Containers/Expressions/TypeTraits.h \
    /home/lali/.local/include/TNL/Algorithms/fill.h \
    /home/lali/.local/include/TNL/Algorithms/detail/Fill.h \
    /home/lali/.local/include/TNL/Algorithms/detail/Fill.hpp \
    /home/lali/.local/include/TNL/Containers/detail/ArrayIO.h \
    /home/lali/.local/include/TNL/Object.h \
    /home/lali/.local/include/TNL/Object.hpp \
    /usr/include/c++/11/cstring \
    /home/lali/.local/include/TNL/Containers/detail/ArrayAssignment.h \
    /home/lali/.local/include/TNL/Containers/detail/MemoryOperations.h \
    /home/lali/.local/include/TNL/Containers/detail/MemoryOperationsSequential.hpp \
    /home/lali/.local/include/TNL/Containers/detail/MemoryOperationsHost.hpp \
    /home/lali/.local/include/TNL/Containers/detail/MemoryOperationsCuda.hpp \
    /home/lali/.local/include/TNL/Containers/Array.hpp

/usr/include/stdc-predef.h:

/usr/local/cuda/bin/../targets/x86_64-linux/include/cuda_runtime.h:

/usr/local/cuda/bin/../targets/x86_64-linux/include/crt/host_config.h:

/usr/include/features.h:

/usr/include/features-time64.h:

/usr/include/x86_64-linux-gnu/bits/wordsize.h:

/usr/include/x86_64-linux-gnu/bits/timesize.h:

/usr/include/x86_64-linux-gnu/sys/cdefs.h:

/usr/include/x86_64-linux-gnu/bits/long-double.h:

/usr/include/x86_64-linux-gnu/gnu/stubs.h:

/usr/include/x86_64-linux-gnu/gnu/stubs-64.h:

/usr/local/cuda/bin/../targets/x86_64-linux/include/builtin_types.h:

/usr/local/cuda/bin/../targets/x86_64-linux/include/device_types.h:

/usr/local/cuda/bin/../targets/x86_64-linux/include/crt/host_defines.h:

/usr/local/cuda/bin/../targets/x86_64-linux/include/driver_types.h:

/usr/local/cuda/bin/../targets/x86_64-linux/include/vector_types.h:

/usr/lib/gcc/x86_64-linux-gnu/11/include/limits.h:

/usr/lib/gcc/x86_64-linux-gnu/11/include/syslimits.h:

/usr/include/limits.h:

/usr/include/x86_64-linux-gnu/bits/libc-header-start.h:

/usr/include/x86_64-linux-gnu/bits/posix1_lim.h:

/usr/include/x86_64-linux-gnu/bits/local_lim.h:

/usr/include/linux/limits.h:

/usr/include/x86_64-linux-gnu/bits/pthread_stack_min-dynamic.h:

/usr/include/x86_64-linux-gnu/bits/posix2_lim.h:

/usr/include/x86_64-linux-gnu/bits/xopen_lim.h:

/usr/include/x86_64-linux-gnu/bits/uio_lim.h:

/usr/lib/gcc/x86_64-linux-gnu/11/include/stddef.h:

/usr/local/cuda/bin/../targets/x86_64-linux/include/surface_types.h:

/usr/local/cuda/bin/../targets/x86_64-linux/include/texture_types.h:

/usr/local/cuda/bin/../targets/x86_64-linux/include/library_types.h:

/usr/local/cuda/bin/../targets/x86_64-linux/include/channel_descriptor.h:

/usr/local/cuda/bin/../targets/x86_64-linux/include/cuda_runtime_api.h:

/usr/local/cuda/bin/../targets/x86_64-linux/include/cuda_device_runtime_api.h:

/usr/include/c++/11/stdlib.h:

/usr/include/c++/11/cstdlib:

/usr/include/x86_64-linux-gnu/c++/11/bits/c++config.h:

/usr/include/x86_64-linux-gnu/c++/11/bits/os_defines.h:

/usr/include/x86_64-linux-gnu/c++/11/bits/cpu_defines.h:

/usr/include/c++/11/pstl/pstl_config.h:

/usr/include/stdlib.h:

/usr/include/x86_64-linux-gnu/bits/waitflags.h:

/usr/include/x86_64-linux-gnu/bits/waitstatus.h:

/usr/include/x86_64-linux-gnu/bits/floatn.h:

/usr/include/x86_64-linux-gnu/bits/floatn-common.h:

/usr/include/x86_64-linux-gnu/bits/types/locale_t.h:

/usr/include/x86_64-linux-gnu/bits/types/__locale_t.h:

/usr/include/x86_64-linux-gnu/sys/types.h:

/usr/include/x86_64-linux-gnu/bits/types.h:

/usr/include/x86_64-linux-gnu/bits/typesizes.h:

/usr/include/x86_64-linux-gnu/bits/time64.h:

/usr/include/x86_64-linux-gnu/bits/types/clock_t.h:

/usr/include/x86_64-linux-gnu/bits/types/clockid_t.h:

/usr/include/x86_64-linux-gnu/bits/types/time_t.h:

/usr/include/x86_64-linux-gnu/bits/types/timer_t.h:

/usr/include/x86_64-linux-gnu/bits/stdint-intn.h:

/usr/include/endian.h:

/usr/include/x86_64-linux-gnu/bits/endian.h:

/usr/include/x86_64-linux-gnu/bits/endianness.h:

/usr/include/x86_64-linux-gnu/bits/byteswap.h:

/usr/include/x86_64-linux-gnu/bits/uintn-identity.h:

/usr/include/x86_64-linux-gnu/sys/select.h:

/usr/include/x86_64-linux-gnu/bits/select.h:

/usr/include/x86_64-linux-gnu/bits/types/sigset_t.h:

/usr/include/x86_64-linux-gnu/bits/types/__sigset_t.h:

/usr/include/x86_64-linux-gnu/bits/types/struct_timeval.h:

/usr/include/x86_64-linux-gnu/bits/types/struct_timespec.h:

/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h:

/usr/include/x86_64-linux-gnu/bits/thread-shared-types.h:

/usr/include/x86_64-linux-gnu/bits/pthreadtypes-arch.h:

/usr/include/x86_64-linux-gnu/bits/atomic_wide_counter.h:

/usr/include/x86_64-linux-gnu/bits/struct_mutex.h:

/usr/include/x86_64-linux-gnu/bits/struct_rwlock.h:

/usr/include/alloca.h:

/usr/include/x86_64-linux-gnu/bits/stdlib-float.h:

/usr/include/c++/11/bits/std_abs.h:

/usr/local/cuda/bin/../targets/x86_64-linux/include/driver_functions.h:

/usr/local/cuda/bin/../targets/x86_64-linux/include/vector_functions.h:

/usr/local/cuda/bin/../targets/x86_64-linux/include/vector_functions.hpp:

/usr/local/cuda/bin/../targets/x86_64-linux/include/crt/common_functions.h:

/usr/include/string.h:

/usr/include/strings.h:

/usr/include/time.h:

/usr/include/x86_64-linux-gnu/bits/time.h:

/usr/include/x86_64-linux-gnu/bits/timex.h:

/usr/include/x86_64-linux-gnu/bits/types/struct_tm.h:

/usr/include/x86_64-linux-gnu/bits/types/struct_itimerspec.h:

/usr/include/c++/11/new:

/usr/include/c++/11/bits/exception.h:

/usr/include/stdio.h:

/usr/lib/gcc/x86_64-linux-gnu/11/include/stdarg.h:

/usr/include/x86_64-linux-gnu/bits/types/__fpos_t.h:

/usr/include/x86_64-linux-gnu/bits/types/__mbstate_t.h:

/usr/include/x86_64-linux-gnu/bits/types/__fpos64_t.h:

/usr/include/x86_64-linux-gnu/bits/types/__FILE.h:

/usr/include/x86_64-linux-gnu/bits/types/FILE.h:

/usr/include/x86_64-linux-gnu/bits/types/struct_FILE.h:

/usr/include/x86_64-linux-gnu/bits/types/cookie_io_functions_t.h:

/usr/include/x86_64-linux-gnu/bits/stdio_lim.h:

/usr/include/assert.h:

/usr/local/cuda/bin/../targets/x86_64-linux/include/crt/math_functions.h:

/usr/include/c++/11/math.h:

/usr/include/c++/11/cmath:

/usr/include/c++/11/bits/cpp_type_traits.h:

/usr/include/c++/11/ext/type_traits.h:

/usr/include/math.h:

/usr/include/x86_64-linux-gnu/bits/math-vector.h:

/usr/include/x86_64-linux-gnu/bits/libm-simd-decl-stubs.h:

/usr/include/x86_64-linux-gnu/bits/flt-eval-method.h:

/usr/include/x86_64-linux-gnu/bits/fp-logb.h:

/usr/include/x86_64-linux-gnu/bits/fp-fast.h:

/usr/include/x86_64-linux-gnu/bits/mathcalls-helper-functions.h:

/usr/include/x86_64-linux-gnu/bits/mathcalls.h:

/usr/include/x86_64-linux-gnu/bits/mathcalls-narrow.h:

/usr/include/x86_64-linux-gnu/bits/iscanonical.h:

/usr/include/c++/11/bits/specfun.h:

/usr/include/c++/11/bits/stl_algobase.h:

/usr/include/c++/11/bits/functexcept.h:

/usr/include/c++/11/bits/exception_defines.h:

/usr/include/c++/11/ext/numeric_traits.h:

/usr/include/c++/11/bits/stl_pair.h:

/usr/include/c++/11/bits/move.h:

/usr/include/c++/11/type_traits:

/usr/include/c++/11/bits/stl_iterator_base_types.h:

/usr/include/c++/11/bits/stl_iterator_base_funcs.h:

/usr/include/c++/11/bits/concept_check.h:

/usr/include/c++/11/debug/assertions.h:

/usr/include/c++/11/bits/stl_iterator.h:

/usr/include/c++/11/bits/ptr_traits.h:

/usr/include/c++/11/debug/debug.h:

/usr/include/c++/11/bits/predefined_ops.h:

/usr/include/c++/11/limits:

/usr/include/c++/11/tr1/gamma.tcc:

/usr/include/c++/11/tr1/special_function_util.h:

/usr/include/c++/11/tr1/bessel_function.tcc:

/usr/include/c++/11/tr1/beta_function.tcc:

/usr/include/c++/11/tr1/ell_integral.tcc:

/usr/include/c++/11/tr1/exp_integral.tcc:

/usr/include/c++/11/tr1/hypergeometric.tcc:

/usr/include/c++/11/tr1/legendre_function.tcc:

/usr/include/c++/11/tr1/modified_bessel_func.tcc:

/usr/include/c++/11/tr1/poly_hermite.tcc:

/usr/include/c++/11/tr1/poly_laguerre.tcc:

/usr/include/c++/11/tr1/riemann_zeta.tcc:

/usr/local/cuda/bin/../targets/x86_64-linux/include/crt/math_functions.hpp:

/usr/local/cuda/bin/../targets/x86_64-linux/include/crt/device_functions.h:

/usr/local/cuda/bin/../targets/x86_64-linux/include/crt/device_functions.hpp:

/usr/local/cuda/bin/../targets/x86_64-linux/include/device_atomic_functions.h:

/usr/local/cuda/bin/../targets/x86_64-linux/include/device_atomic_functions.hpp:

/usr/local/cuda/bin/../targets/x86_64-linux/include/crt/device_double_functions.h:

/usr/local/cuda/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp:

/usr/local/cuda/bin/../targets/x86_64-linux/include/sm_20_atomic_functions.h:

/usr/local/cuda/bin/../targets/x86_64-linux/include/sm_20_atomic_functions.hpp:

/usr/local/cuda/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.h:

/usr/local/cuda/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.hpp:

/usr/local/cuda/bin/../targets/x86_64-linux/include/sm_35_atomic_functions.h:

/usr/local/cuda/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h:

/usr/local/cuda/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.hpp:

/usr/local/cuda/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h:

/usr/local/cuda/bin/../targets/x86_64-linux/include/sm_20_intrinsics.hpp:

/usr/local/cuda/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h:

/usr/local/cuda/bin/../targets/x86_64-linux/include/sm_30_intrinsics.hpp:

/usr/local/cuda/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h:

/usr/local/cuda/bin/../targets/x86_64-linux/include/sm_32_intrinsics.hpp:

/usr/local/cuda/bin/../targets/x86_64-linux/include/sm_35_intrinsics.h:

/usr/local/cuda/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h:

/usr/local/cuda/bin/../targets/x86_64-linux/include/sm_61_intrinsics.hpp:

/usr/local/cuda/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h:

/usr/local/cuda/bin/../targets/x86_64-linux/include/crt/sm_70_rt.hpp:

/usr/local/cuda/bin/../targets/x86_64-linux/include/crt/sm_80_rt.h:

/usr/local/cuda/bin/../targets/x86_64-linux/include/crt/sm_80_rt.hpp:

/usr/local/cuda/bin/../targets/x86_64-linux/include/crt/sm_90_rt.h:

/usr/local/cuda/bin/../targets/x86_64-linux/include/crt/sm_90_rt.hpp:

/usr/local/cuda/bin/../targets/x86_64-linux/include/texture_indirect_functions.h:

/usr/local/cuda/bin/../targets/x86_64-linux/include/surface_indirect_functions.h:

/usr/local/cuda/bin/../targets/x86_64-linux/include/crt/cudacc_ext.h:

/usr/local/cuda/bin/../targets/x86_64-linux/include/device_launch_parameters.h:

/usr/include/c++/11/functional:

/usr/include/c++/11/bits/stl_function.h:

/usr/include/c++/11/backward/binders.h:

/usr/include/c++/11/tuple:

/usr/include/c++/11/utility:

/usr/include/c++/11/bits/stl_relops.h:

/usr/include/c++/11/initializer_list:

/usr/include/c++/11/array:

/usr/include/c++/11/bits/range_access.h:

/usr/include/c++/11/bits/uses_allocator.h:

/usr/include/c++/11/bits/invoke.h:

/usr/include/c++/11/bits/functional_hash.h:

/usr/include/c++/11/bits/hash_bytes.h:

/usr/include/c++/11/bits/refwrap.h:

/usr/include/c++/11/bits/std_function.h:

/usr/include/c++/11/typeinfo:

/usr/include/c++/11/unordered_map:

/usr/include/c++/11/bits/allocator.h:

/usr/include/x86_64-linux-gnu/c++/11/bits/c++allocator.h:

/usr/include/c++/11/ext/new_allocator.h:

/usr/include/c++/11/bits/memoryfwd.h:

/usr/include/c++/11/ext/alloc_traits.h:

/usr/include/c++/11/bits/alloc_traits.h:

/usr/include/c++/11/bits/stl_construct.h:

/usr/include/c++/11/ext/aligned_buffer.h:

/usr/include/c++/11/bits/hashtable.h:

/usr/include/c++/11/bits/hashtable_policy.h:

/usr/include/c++/11/bits/enable_special_members.h:

/usr/include/c++/11/bits/node_handle.h:

/usr/include/c++/11/bits/unordered_map.h:

/usr/include/c++/11/bits/erase_if.h:

/usr/include/c++/11/vector:

/usr/include/c++/11/bits/stl_uninitialized.h:

/usr/include/c++/11/bits/stl_vector.h:

/usr/include/c++/11/bits/stl_bvector.h:

/usr/include/c++/11/bits/vector.tcc:

/usr/include/c++/11/bits/stl_algo.h:

/usr/include/c++/11/bits/algorithmfwd.h:

/usr/include/c++/11/bits/stl_heap.h:

/usr/include/c++/11/bits/stl_tempbuf.h:

/usr/include/c++/11/bits/uniform_int_dist.h:

/usr/include/c++/11/iostream:

/usr/include/c++/11/ostream:

/usr/include/c++/11/ios:

/usr/include/c++/11/iosfwd:

/usr/include/c++/11/bits/stringfwd.h:

/usr/include/c++/11/bits/postypes.h:

/usr/include/c++/11/cwchar:

/usr/include/wchar.h:

/usr/include/x86_64-linux-gnu/bits/wchar.h:

/usr/include/x86_64-linux-gnu/bits/types/wint_t.h:

/usr/include/x86_64-linux-gnu/bits/types/mbstate_t.h:

/usr/include/c++/11/exception:

/usr/include/c++/11/bits/exception_ptr.h:

/usr/include/c++/11/bits/cxxabi_init_exception.h:

/usr/include/c++/11/bits/nested_exception.h:

/usr/include/c++/11/bits/char_traits.h:

/usr/include/c++/11/cstdint:

/usr/lib/gcc/x86_64-linux-gnu/11/include/stdint.h:

/usr/include/stdint.h:

/usr/include/x86_64-linux-gnu/bits/stdint-uintn.h:

/usr/include/c++/11/bits/localefwd.h:

/usr/include/x86_64-linux-gnu/c++/11/bits/c++locale.h:

/usr/include/c++/11/clocale:

/usr/include/locale.h:

/usr/include/x86_64-linux-gnu/bits/locale.h:

/usr/include/c++/11/cctype:

/usr/include/ctype.h:

/usr/include/c++/11/bits/ios_base.h:

/usr/include/c++/11/ext/atomicity.h:

/usr/include/x86_64-linux-gnu/c++/11/bits/gthr.h:

/usr/include/x86_64-linux-gnu/c++/11/bits/gthr-default.h:

/usr/include/pthread.h:

/usr/include/sched.h:

/usr/include/x86_64-linux-gnu/bits/sched.h:

/usr/include/x86_64-linux-gnu/bits/types/struct_sched_param.h:

/usr/include/x86_64-linux-gnu/bits/cpu-set.h:

/usr/include/x86_64-linux-gnu/bits/setjmp.h:

/usr/include/x86_64-linux-gnu/bits/types/struct___jmp_buf_tag.h:

/usr/include/x86_64-linux-gnu/c++/11/bits/atomic_word.h:

/usr/include/x86_64-linux-gnu/sys/single_threaded.h:

/usr/include/c++/11/bits/locale_classes.h:

/usr/include/c++/11/string:

/usr/include/c++/11/bits/ostream_insert.h:

/usr/include/c++/11/bits/cxxabi_forced.h:

/usr/include/c++/11/bits/basic_string.h:

/usr/include/c++/11/string_view:

/usr/include/c++/11/bits/string_view.tcc:

/usr/include/c++/11/ext/string_conversions.h:

/usr/include/c++/11/cstdio:

/usr/include/c++/11/cerrno:

/usr/include/errno.h:

/usr/include/x86_64-linux-gnu/bits/errno.h:

/usr/include/linux/errno.h:

/usr/include/x86_64-linux-gnu/asm/errno.h:

/usr/include/asm-generic/errno.h:

/usr/include/asm-generic/errno-base.h:

/usr/include/x86_64-linux-gnu/bits/types/error_t.h:

/usr/include/c++/11/bits/charconv.h:

/usr/include/c++/11/bits/basic_string.tcc:

/usr/include/c++/11/bits/locale_classes.tcc:

/usr/include/c++/11/system_error:

/usr/include/x86_64-linux-gnu/c++/11/bits/error_constants.h:

/usr/include/c++/11/stdexcept:

/usr/include/c++/11/streambuf:

/usr/include/c++/11/bits/streambuf.tcc:

/usr/include/c++/11/bits/basic_ios.h:

/usr/include/c++/11/bits/locale_facets.h:

/usr/include/c++/11/cwctype:

/usr/include/wctype.h:

/usr/include/x86_64-linux-gnu/bits/wctype-wchar.h:

/usr/include/x86_64-linux-gnu/c++/11/bits/ctype_base.h:

/usr/include/c++/11/bits/streambuf_iterator.h:

/usr/include/x86_64-linux-gnu/c++/11/bits/ctype_inline.h:

/usr/include/c++/11/bits/locale_facets.tcc:

/usr/include/c++/11/bits/basic_ios.tcc:

/usr/include/c++/11/bits/ostream.tcc:

/usr/include/c++/11/istream:

/usr/include/c++/11/bits/istream.tcc:

/home/lali/.local/include/TNL/Containers/Array.h:

/usr/include/c++/11/list:

/usr/include/c++/11/bits/stl_list.h:

/usr/include/c++/11/bits/allocated_ptr.h:

/usr/include/c++/11/bits/list.tcc:

/home/lali/.local/include/TNL/File.h:

/usr/include/c++/11/fstream:

/usr/include/c++/11/bits/codecvt.h:

/usr/include/x86_64-linux-gnu/c++/11/bits/basic_file.h:

/usr/include/x86_64-linux-gnu/c++/11/bits/c++io.h:

/usr/include/c++/11/bits/fstream.tcc:

/home/lali/.local/include/TNL/String.h:

/usr/include/c++/11/sstream:

/usr/include/c++/11/bits/sstream.tcc:

/home/lali/.local/include/TNL/String.hpp:

/home/lali/.local/include/TNL/Assert.h:

/home/lali/.local/include/TNL/Cuda/CudaCallable.h:

/home/lali/.local/include/TNL/Math.h:

/usr/include/c++/11/algorithm:

/usr/include/c++/11/pstl/glue_algorithm_defs.h:

/usr/include/c++/11/pstl/execution_defs.h:

/home/lali/.local/include/TNL/TypeTraits.h:

/home/lali/.local/include/TNL/Allocators/Host.h:

/usr/include/c++/11/memory:

/usr/include/c++/11/bits/stl_raw_storage_iter.h:

/usr/include/c++/11/bits/align.h:

/usr/include/c++/11/bit:

/usr/include/c++/11/bits/unique_ptr.h:

/usr/include/c++/11/bits/shared_ptr.h:

/usr/include/c++/11/bits/shared_ptr_base.h:

/usr/include/c++/11/ext/concurrence.h:

/usr/include/c++/11/bits/shared_ptr_atomic.h:

/usr/include/c++/11/bits/atomic_base.h:

/usr/include/c++/11/bits/atomic_lockfree_defines.h:

/usr/include/c++/11/backward/auto_ptr.h:

/usr/include/c++/11/pstl/glue_memory_defs.h:

/home/lali/.local/include/TNL/Allocators/Cuda.h:

/home/lali/.local/include/TNL/Exceptions/CudaBadAlloc.h:

/home/lali/.local/include/TNL/Exceptions/CudaSupportMissing.h:

/home/lali/.local/include/TNL/Cuda/CheckDevice.h:

/home/lali/.local/include/TNL/Exceptions/CudaRuntimeError.h:

/home/lali/.local/include/TNL/File.hpp:

/home/lali/.local/include/TNL/Cuda/LaunchHelpers.h:

/home/lali/.local/include/TNL/DiscreteMath.h:

/home/lali/.local/include/TNL/Cuda/DummyDefs.h:

/home/lali/.local/include/TNL/Exceptions/FileSerializationError.h:

/home/lali/.local/include/TNL/Exceptions/FileDeserializationError.h:

/home/lali/.local/include/TNL/Exceptions/NotImplementedError.h:

/home/lali/.local/include/TNL/Allocators/Default.h:

/home/lali/.local/include/TNL/Devices/Sequential.h:

/home/lali/.local/include/TNL/Devices/Host.h:

/home/lali/.local/include/TNL/Config/ConfigDescription.h:

/home/lali/.local/include/TNL/Config/ConfigEntry.h:

/usr/include/c++/11/optional:

/home/lali/.local/include/TNL/Config/ConfigEntryBase.h:

/home/lali/.local/include/TNL/Config/ConfigEntryType.h:

/usr/include/c++/11/variant:

/usr/include/c++/11/bits/parse_numbers.h:

/home/lali/.local/include/TNL/Config/ConfigEntryList.h:

/home/lali/.local/include/TNL/Config/ConfigDelimiter.h:

/home/lali/.local/include/TNL/Exceptions/ConfigError.h:

/home/lali/.local/include/TNL/Config/ParameterContainer.h:

/home/lali/.local/include/TNL/TypeInfo.h:

/usr/include/c++/11/cxxabi.h:

/usr/include/x86_64-linux-gnu/c++/11/bits/cxxabi_tweaks.h:

/usr/lib/gcc/x86_64-linux-gnu/11/include/omp.h:

/home/lali/.local/include/TNL/Devices/Cuda.h:

/home/lali/.local/include/TNL/Cuda/KernelLaunch.h:

/home/lali/.local/include/TNL/Containers/ArrayView.h:

/home/lali/.local/include/TNL/Containers/ArrayView.hpp:

/home/lali/.local/include/TNL/Algorithms/copy.h:

/home/lali/.local/include/TNL/Algorithms/detail/Copy.h:

/home/lali/.local/include/TNL/Algorithms/detail/Copy.hpp:

/home/lali/.local/include/TNL/Algorithms/parallelFor.h:

/home/lali/.local/include/TNL/Algorithms/detail/ParallelFor1D.h:

/home/lali/.local/include/TNL/Cuda/DeviceInfo.h:

/home/lali/.local/include/TNL/Cuda/DeviceInfo.hpp:

/home/lali/.local/include/TNL/Algorithms/detail/ParallelFor2D.h:

/home/lali/.local/include/TNL/Algorithms/detail/ParallelFor3D.h:

/home/lali/.local/include/TNL/Algorithms/equal.h:

/home/lali/.local/include/TNL/Algorithms/detail/Equal.h:

/home/lali/.local/include/TNL/Algorithms/detail/Equal.hpp:

/home/lali/.local/include/TNL/Algorithms/reduce.h:

/home/lali/.local/include/TNL/Functional.h:

/home/lali/.local/include/TNL/Algorithms/detail/Reduction.h:

/home/lali/.local/include/TNL/Algorithms/detail/Reduction.hpp:

/home/lali/.local/include/TNL/Algorithms/detail/CudaReductionKernel.h:

/home/lali/.local/include/TNL/Algorithms/CudaReductionBuffer.h:

/home/lali/.local/include/TNL/Containers/Expressions/TypeTraits.h:

/home/lali/.local/include/TNL/Algorithms/fill.h:

/home/lali/.local/include/TNL/Algorithms/detail/Fill.h:

/home/lali/.local/include/TNL/Algorithms/detail/Fill.hpp:

/home/lali/.local/include/TNL/Containers/detail/ArrayIO.h:

/home/lali/.local/include/TNL/Object.h:

/home/lali/.local/include/TNL/Object.hpp:

/usr/include/c++/11/cstring:

/home/lali/.local/include/TNL/Containers/detail/ArrayAssignment.h:

/home/lali/.local/include/TNL/Containers/detail/MemoryOperations.h:

/home/lali/.local/include/TNL/Containers/detail/MemoryOperationsSequential.hpp:

/home/lali/.local/include/TNL/Containers/detail/MemoryOperationsHost.hpp:

/home/lali/.local/include/TNL/Containers/detail/MemoryOperationsCuda.hpp:

/home/lali/.local/include/TNL/Containers/Array.hpp:
