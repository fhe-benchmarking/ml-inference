func.func @approx_sign(%x: tensor<1x1024xf32>) -> tensor<1x1024xf32> {
  %c11 = arith.constant dense<-260.03867215588> : tensor<1x1024xf32>
  %c9 = arith.constant dense<746.781707684981> : tensor<1x1024xf32>
  %c7 = arith.constant dense<-797.090149675776> : tensor<1x1024xf32>
  %c5 = arith.constant dense<388.964712077092> : tensor<1x1024xf32>
  %c3 = arith.constant dense<-86.6415008377027> : tensor<1x1024xf32>
  %c1 = arith.constant dense<8.82341343192733> : tensor<1x1024xf32>

  %x2 = arith.mulf %x, %x : tensor<1x1024xf32>
  %x3 = arith.mulf %x2, %x : tensor<1x1024xf32>
  %x4 = arith.mulf %x2, %x2 : tensor<1x1024xf32>
  %x5 = arith.mulf %x4, %x : tensor<1x1024xf32>
  %x6 = arith.mulf %x4, %x2 : tensor<1x1024xf32>
  %x7 = arith.mulf %x6, %x : tensor<1x1024xf32>
  %x8 = arith.mulf %x4, %x4 : tensor<1x1024xf32>
  %x9 = arith.mulf %x8, %x : tensor<1x1024xf32>
  %x11 = arith.mulf %x5, %x6 : tensor<1x1024xf32>

  %s1 = arith.mulf %x, %c1 : tensor<1x1024xf32>
  %s3 = arith.mulf %x3, %c3 : tensor<1x1024xf32>
  %s5 = arith.mulf %x5, %c5 : tensor<1x1024xf32>
  %s7 = arith.mulf %x7, %c7 : tensor<1x1024xf32>
  %s9 = arith.mulf %x9, %c9 : tensor<1x1024xf32>
  %s11 = arith.mulf %x11, %c11 : tensor<1x1024xf32>

  %sum1 = arith.addf %s1, %s3 : tensor<1x1024xf32>
  %sum2 = arith.addf %sum1, %s5 : tensor<1x1024xf32>
  %sum3 = arith.addf %sum2, %s7 : tensor<1x1024xf32>
  %sum4 = arith.addf %sum3, %s9 : tensor<1x1024xf32>
  %sum5 = arith.addf %sum4, %s11 : tensor<1x1024xf32>
  return %sum5 : tensor<1x1024xf32>
}

func.func @approx_relu(%x: tensor<1x1024xf32>) -> tensor<1x1024xf32> {
  %sign = call @approx_sign(%x) : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
  %signed = arith.mulf %sign, %x : tensor<1x1024xf32>
  %sum = arith.addf %signed, %x : tensor<1x1024xf32>
  %c0_5 = arith.constant dense<0.5> : tensor<1x1024xf32>
  %norm = arith.mulf %sum, %c0_5 : tensor<1x1024xf32>
  return %norm : tensor<1x1024xf32>
}

func.func @mlp(%input: tensor<1x1024xf32>, %fc1: tensor<1024x1024xf32>, %fc2: tensor<1024x1024xf32>, %fc1_buffer: tensor<1x1024xf32>, %fc2_buffer: tensor<1x1024xf32>) -> tensor<1x1024xf32> attributes {llvm.emit_c_interface} {
  %fc1_result = linalg.matmul ins(%input, %fc1 : tensor<1x1024xf32>, tensor<1024x1024xf32>) outs(%fc1_buffer : tensor<1x1024xf32>) -> tensor<1x1024xf32>
  %relu1 = call @approx_relu(%fc1_result) : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
  %fc2_result = linalg.matmul ins(%relu1, %fc2 : tensor<1x1024xf32>, tensor<1024x1024xf32>) outs(%fc2_buffer : tensor<1x1024xf32>) -> tensor<1x1024xf32>
  return %fc2_result : tensor<1x1024xf32>
}