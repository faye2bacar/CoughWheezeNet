package com.example.coughwheeze.audio

import android.media.*

class AudioRecorder(private val sr: Int = 16000) {
  private val minBuf = AudioRecord.getMinBufferSize(sr,
    AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
  private val rec = AudioRecord(
    MediaRecorder.AudioSource.MIC, sr,
    AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, minBuf
  )
  fun start() = rec.startRecording()
  fun stop() = rec.stop()
  fun read(buf: ShortArray): Int = rec.read(buf, 0, buf.size)
}
