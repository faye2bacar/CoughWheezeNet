package com.example.coughwheeze.ml

import android.content.Context
import java.io.File
import java.net.URL
import java.security.MessageDigest

class ModelUpdater(private val ctx: Context) {
  private val modelsDir = File(ctx.filesDir, "models").apply { mkdirs() }
  private val versionFile = File(modelsDir, "version.txt")

  fun localVersion(): String = if (versionFile.exists()) versionFile.readText() else "0.0.0"
  fun setLocalVersion(v: String) = versionFile.writeText(v)

  fun updateIfNeeded(manifestUrl: String, baseUrl: String): ModelManifest? {
    val remote = URL(manifestUrl).readText()
    val man = ModelManifest.parse(remote)
    if (man.version <= localVersion()) return man
    man.models.forEach { m ->
      val bytes = URL("$baseUrl/${m.file}").readBytes()
      val ok = verifySha256(bytes, m.hash.removePrefix("sha256:"))
      require(ok) { "Hash mismatch for ${m.file}" }
      File(modelsDir, m.file).writeBytes(bytes)
    }
    setLocalVersion(man.version)
    File(modelsDir, "manifest.json").writeText(remote)
    return man
  }

  fun modelFile(name: String): File = File(modelsDir, "$name.tflite")

  private fun verifySha256(bytes: ByteArray, hex: String): Boolean {
    val md = MessageDigest.getInstance("SHA-256")
    val dig = md.digest(bytes).joinToString("") { "%02x".format(it) }
    return dig.equals(hex, ignoreCase = true)
  }
}
