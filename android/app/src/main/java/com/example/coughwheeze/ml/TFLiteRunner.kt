package com.example.coughwheeze.ml

import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class TFLiteRunner(modelPath: String) {
  private val tflite: Interpreter
  init {
    val file = File(modelPath)
    val mbb: MappedByteBuffer = FileInputStream(file).channel
      .map(FileChannel.MapMode.READ_ONLY, 0, file.length())
    tflite = Interpreter(mbb)
  }
  fun run(input: Array<Array<Array<FloatArray>>>): Pair<FloatArray, FloatArray> {
    val logits = FloatArray(3)
    val exac = FloatArray(input[0][0].size)
    val outputs = hashMapOf(0 to logits, 1 to exac)
    tflite.runForMultipleInputsOutputs(arrayOf(input), outputs)
    return logits to exac
  }
}
