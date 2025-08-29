package com.example.coughwheeze.ml

import org.json.JSONObject

data class ModelInfo(
  val name: String,
  val file: String,
  val hash: String
)

data class Preproc(
  val sr: Int, val n_fft: Int, val win_length: Int,
  val hop_length: Int, val n_mels: Int, val fmin: Int, val fmax: Int
)

data class ModelManifest(
  val version: String,
  val models: List<ModelInfo>,
  val preproc: Preproc
) {
  companion object {
    fun parse(json: String): ModelManifest {
      val root = JSONObject(json)
      val version = root.getString("version")
      val pre = root.getJSONObject("preproc")
      val preproc = Preproc(
        pre.getInt("sr"), pre.getInt("n_fft"), pre.getInt("win_length"),
        pre.getInt("hop_length"), pre.getInt("n_mels"), pre.getInt("fmin"), pre.getInt("fmax")
      )
      val arr = root.getJSONArray("models")
      val models = buildList {
        for (i in 0 until arr.length()) {
          val m = arr.getJSONObject(i)
          add(ModelInfo(m.getString("name"), m.getString("file"), m.getString("hash")))
        }
      }
      return ModelManifest(version, models, preproc)
    }
  }
}
