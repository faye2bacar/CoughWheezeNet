package com.example.coughwheeze

import android.Manifest
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.example.coughwheeze.ml.ModelUpdater
import com.example.coughwheeze.ml.TFLiteRunner
import kotlin.math.exp

class MainActivity: ComponentActivity() {
  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    val reqPerm = registerForActivityResult(
      ActivityResultContracts.RequestPermission()
    ) { }
    reqPerm.launch(Manifest.permission.RECORD_AUDIO)

    setContent {
      MaterialTheme {
        AppScreen()
      }
    }
  }
}

@Composable
fun AppScreen() {
  val ctx = androidx.compose.ui.platform.LocalContext.current
  val updater = remember { ModelUpdater(ctx) }
  var version by remember { mutableStateOf(updater.localVersion()) }
  var status by remember { mutableStateOf("Ready") }
  var probs by remember { mutableStateOf(floatArrayOf(0f,0f,1f)) }

  Column(Modifier.fillMaxSize().padding(16.dp), horizontalAlignment = Alignment.CenterHorizontally) {
    Text("Cough+Wheeze Net", style = MaterialTheme.typography.titleLarge)
    Spacer(Modifier.height(12.dp))
    Text("Model version: " + version)
    Spacer(Modifier.height(12.dp))
    Row {
      Button(onClick = {
        try {
          val man = updater.updateIfNeeded(
            manifestUrl = "https://<blob>/manifest.json",
            baseUrl = "https://<blob>"
          )
          version = updater.localVersion()
          status = "Updated to " + (man?.version ?: version)
        } catch (e: Exception) { status = "Update failed: " + e.message }
      }) { Text("Check updates") }
    }
    Spacer(Modifier.height(16.dp))
    Button(onClick = {
      try {
        val model = updater.modelFile("crnn")
        val runner = TFLiteRunner(model.absolutePath)
        val T = 300; val M = 64
        val input = Array(1) { Array(1) { Array(T) { FloatArray(M) } } }
        val res = runner.run(input)
        val logits = res.first
        probs = softmax(logits)
        status = "Inference OK"
      } catch (e: Exception) { status = "Inference error: " + e.message }
    }) { Text("Run demo inference") }
    Spacer(Modifier.height(12.dp))
    Text("asthme=%.2f  COPD=%.2f  sain=%.2f".format(probs[0], probs[1], probs[2]))
    Spacer(Modifier.height(6.dp))
    Text(status)
  }
}

fun softmax(logits: FloatArray): FloatArray {
  val max = logits.maxOrNull() ?: 0f
  val exps = logits.map { exp((it - max).toDouble()).toFloat() }
  val sum = exps.sum()
  return exps.map { it/sum }.toFloatArray()
}
