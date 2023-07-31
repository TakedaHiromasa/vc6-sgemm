#####################################################################
# 16個のラッキー7をQPUからHostに出力するサンプルプログラム
# TMUを使った[レジスタ]->[GPUメモリ]転送が確認できる
#####################################################################
import numpy as np

from videocore6.assembler import qpu
from videocore6.driver import Driver

@qpu
def output_test(asm):
    params = [
        'IN',
        'OUT'
    ]

    g = globals()
    for i, reg in enumerate(params):
        g['reg_' + reg] = g['rf' + str(i+32)]

    # uniformから値を取り出す
    nop(sig=ldunifrf(r0))  # unif_params.addresses()[0,0]

    # QPU番号取得
    tidx(r1)
    shr(r3, r1, 2)
    band(r3, r3, 0b1111)  # qpu_id

    nop(sig=ldunifrf(r1))  # unif_params.shape[1]
    shl(r1, r1, 2)         # param_size = len(param)*4
    umul24(r1, r3, r1)
    add(r0, r0, r1)        # param_addr = params.addresses + (qpu_num*param_size)

    # TMU用 Baseアドレス生成
    eidx(r2)        # r0 = [0 ... 15]
    shl(r2, r2, 2)  # 各数値を4倍(float32のバイト数分)
    add(r0, r0, r2)

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


    # データの読み込み
    add(reg_IN, reg_IN, r2)
    mov(tmua, reg_IN, sig = thrsw)
    nop()
    nop()
    nop(sig = ldtmu(r0))

    # element_number
    eidx(r2)         # r2 = [0 ... 15]
    shl(r2, r2, 2)   # 各数値を4倍
    add(reg_OUT, reg_OUT, r2)  # result[] のアドレスから ストライド=4バイトのアドレスベクトルを生成

    mov(tmud, r0)  # 書き出すデータ
    mov(tmua, reg_OUT)  # 書き出し先アドレスベクトル
    tmuwt()

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
    with Driver() as drv:
        # set 1 or 8
        n_threads = 8

        code = drv.program(output_test)

        input  = drv.alloc((n_threads, 16), dtype='uint32')
        for th in range(n_threads):
          input[th,:] = th+1.0

        result = drv.alloc((n_threads, 16), dtype='uint32')
        result[:] = 0.0

        def params(num_th):
            return [
              input.addresses()[num_th, 0],
              result.addresses()[num_th, 0]
            ]

        unif_params = drv.alloc((n_threads, len(params(0))), dtype = 'uint32')
        for th in range(n_threads):
            unif_params[th] = params(th)

        # Allocate uniforms.
        uniforms = drv.alloc(2 , dtype='uint32')
        uniforms[0] = unif_params.addresses()[0,0]
        uniforms[1] = unif_params.shape[1]

        drv.execute(code, uniforms.addresses()[0], thread=n_threads)

        print(result)

if __name__ == '__main__':
    main()