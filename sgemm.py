import numpy as np
import struct
import time

from videocore6 import pack_unpack
from videocore6.assembler import qpu
from videocore6.driver import Driver

# C ← αAB + βC

def astype_int24(array):
    array = np.left_shift(array, 8)
    array = np.right_shift(array, 8)
    return array

@qpu
def kernel(asm, thread):

    params = [
        'P',
        'Q',
        'R',
        'A_base',
        'A_stride',
        'B_base',
        'B_stride',
        'C_base',
        'C_stride',
        'alpha',
        'beta',
    ]

    values = [
        'A_cur',
        'B_cur',
        'C_cur',
        'B_tmp',
        'm',
        'k',
        'n'
    ]

    g = globals()
    for i, reg in enumerate(params + values):
        g['reg_' + reg] = g['rf' + str(i+32)]
        
    # uniformから値を取り出す
    nop(sig=ldunifrf(r0))  # unif_params.addresses()[0,0]

    if thread == 8:
        # QPU番号取得
        tidx(r1)
        shr(r3, r1, 2)
        band(r3, r3, 0b1111)  # qpu_id

        nop(sig=ldunifrf(r1))  # unif_params.shape[1]
        shl(r1, r1, 2)         # param_size = len(param)*4
        umul24(r1, r3, r1)
        add(r0, r0, r1)        # param_addr = params.addresses + (qpu_num*param_size)
    
    # TMU用 Baseアドレス生成
    eidx(r1)        # r0 = [0 ... 15]
    shl(r1, r1, 2)  # 各数値を4倍(float32のバイト数分)
    add(r0, r0, r1)

    # データの読み込み
    # params()の構造体的な塊をTMUでごそっと取ってくる
    mov(tmua, r0, sig = thrsw)
    nop()
    nop()
    nop(sig = ldtmu(r1))

    # ごそっと取ってきた値を各レジスタに振り分け
    regs = [g['reg_' + reg] for reg in params]
    n = len(regs)
    for i in range(n):
      rotate(r5rep, r1, -i)
      mov(regs[i], r5)


    # Baseのアドレスベクトルを生成
    # Aは縦に値を取るため1行飛びの番地を生成
    eidx(r0)
    shl(r1, r0, 2)
    umul24(r3, r0, reg_A_stride)
    add(reg_A_base, reg_A_base, r3)
    add(reg_B_base, reg_B_base, r1)


    # rf[0~31]を初期化
    for i in range(16):
        mov(rf[i], 0.0).mov(rf[i+16], 0.0)


    # m = P/16
    shr(reg_m, reg_P, 4)
    with loop as loop_M:
      # n = R/16
      shr(reg_n, reg_R, 4)

      mov(reg_B_tmp, reg_B_base)
      mov(reg_B_cur, reg_B_base)
      mov(reg_C_cur, reg_C_base)

      with loop as loop_N:
          
          mov(reg_A_cur, reg_A_base)
          mov(reg_k, reg_Q)

          with loop as loop_K:
              # Aから縦に16個取る -> r3
              mov(tmua, reg_A_cur, sig = thrsw)
              nop(sig = ldtmu(r3))

              # Bから横に16個取る -> r4
              mov(tmua, reg_B_cur, sig = thrsw)
              nop(sig = ldtmu(r4))

              for i in range(16):
                  rotate(r5rep, r3, -i) # r3には行列Aからのロード分 
                  fmul(r0, r4, r5)      # r4には行列Bからのロード分 
                  fadd(rf[i], rf[i], r0)

              sub(reg_k, reg_k, 1, cond = 'pushz')
              loop_K.b(cond='anyna')
              add(reg_A_cur, reg_A_cur, 4)            # delay slot
              add(reg_B_cur, reg_B_cur, reg_B_stride) # delay slot
              nop() # delay slot

          eidx(r2)
          shl(r2, r2, 2)  # 各数値を4倍
          add(r0, reg_C_cur, r2) # アドレスベクトルを生成

          mov(r1, r0)
          for i in range(16):
              mov(tmua, r1, sig = thrsw)
              add(r1, r1, reg_C_stride)     # nop()
              fmul(rf[i], rf[i], reg_alpha) # nop()
              nop(sig = ldtmu(r4))
              fmul(r2, r4, reg_beta)
              fadd(rf[i], rf[i], r2)

          mov(r1, r0)
          for i in range(16):
              mov(tmud, rf[i])  # 書き出すデータ
              mov(tmua, r1)     # 書き出し先アドレスベクトル
              add(r1, r1, reg_C_stride)  # addressのインクリメント
              mov(rf[i], 0.0)   # rfの初期化
              tmuwt()

          mov(r0, 2)
          shl(r0, r0, 5) # 64=4(byte)*16
          add(reg_B_tmp, reg_B_tmp, r0)
          mov(reg_B_cur, reg_B_tmp)
          add(reg_C_cur, reg_C_cur, r0)
          
          sub(reg_n, reg_n, 1, cond = 'pushz')
          loop_N.b(cond='anyna')
          nop() # delay slot
          nop() # delay slot
          nop() # delay slot

      shl(r0, reg_A_stride, 4)  # (reg_A_stride)*16
      shl(r1, reg_C_stride, 4)  # (reg_C_stride)*16
      
      sub(reg_m, reg_m, 1, cond = 'pushz')
      loop_M.b(cond='anyna')
      add(reg_A_base, reg_A_base, r0) # delay slot
      add(reg_C_base, reg_C_base, r1) # delay slot
      nop() # delay slot


    # GPUコードを終了する
    # 以下，なんかよくわからないがTMUを使った場合付けないと危ないのでつける
    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def main():

    m = 1024
    k = 1024
    n = 1024

    th_y=2
    th_x=4

    # set 1 or 8
    n_threads = th_y*th_x

    assert m % (16*th_y) == 0
    assert n % (16*th_x) == 0

    with Driver() as drv:
        # Allocate matrices.
        #      K           N            N  
        #    ┌---┐       ┌---┐        ┌---┐
        #  M | A |  x  K | B |  =>  M | C |
        #    └---┘       └---┘        └---┘
        A = drv.alloc((m, k), dtype='float32')
        B = drv.alloc((k, n), dtype='float32')
        C = drv.alloc((m, n), dtype='float32')


        # Initialize matrices.
        np.random.seed(0)
        alpha = np.random.randn()
        beta  = np.random.randn()

        A[:] = np.random.randn(m, k).astype(A.dtype)
        B[:] = np.random.randn(k, n).astype(B.dtype)
        C[:] = np.random.randn(m, n).astype(C.dtype)
        
        
        # Reference
        start = time.time()
        R = alpha * np.dot(A, B) + beta * C
        elapsed_ref = time.time() - start


        def params(num_th):
            block_m = m//th_y
            block_n = n//th_x

            y = block_m * (num_th//th_x)
            x = block_n * (num_th%th_x)

            return [
                block_m, k, block_n,
                A.addresses()[y, 0],
                A.strides[0],
                B.addresses()[0, x],
                B.strides[0],
                C.addresses()[y, x],
                C.strides[0],

                # pythonの数値 -> packでバイト列に変換Cの構造体
                # これが無いとCの構造体互換にならない
                *pack_unpack('f', 'I', [alpha, beta]),
            ]

        unif_params = drv.alloc((n_threads, len(params(0))), dtype = 'uint32')
        for th in range(n_threads):
            unif_params[th] = params(th)

        # Allocate uniforms.
        uniforms = drv.alloc(2 , dtype='uint32')
        uniforms[0] = unif_params.addresses()[0,0]
        uniforms[1] = unif_params.shape[1]


        # Run the program
        code = drv.program(kernel, thread=n_threads)

        start = time.time()
        drv.execute(code, uniforms.addresses()[0], thread=n_threads)
        elapsed_gpu = time.time() - start

        # Show result
        def Gflops(sec):
            return (2*m*k*n + 3*m*n)/sec * 1e-9

        print('==================== Sgemm ====================')
        print('C ← αAB + βC')
        print(f'A = {m}x{k}, B = {k}x{n}, C = {m}x{n}')

        print('\n【Performance】')
        print(f'threads: {n_threads}({th_y}x{th_x})')
        print('numpy: {:.4f} msec, {:.4f} Gflops'.format(elapsed_ref*1000, Gflops(elapsed_ref)))
        print('GPU  : {:.4f} msec, {:.4f} Gflops'.format(elapsed_gpu*1000, Gflops(elapsed_gpu)))

        print('\n【Error】')
        print('minimum absolute error: {:.4e}'.format(float(np.min(np.abs(R - C)))))
        print('maximum absolute error: {:.4e}'.format(float(np.max(np.abs(R - C)))))
        print('minimum relative error: {:.4e}'.format(float(np.min(np.abs((R - C) / R)))))
        print('maximum relative error: {:.4e}'.format(float(np.max(np.abs((R - C) / R)))))

if __name__ == '__main__':
    main()
