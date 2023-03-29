<template>
  <div>
    <video
      ref="videoPlayer"
      class="video-js vjs-luxmty-skin vjs-big-play-centered h-100 position-static"
    ></video>
  </div>
</template>

<style src="video.js/dist/video-js.css"></style>
<style src="@/scss/video-player.scss" lang="scss"></style>

<script>
import videojs from "video.js"
import "videojs-hotkeys"
import "videojs-http-source-selector"
import "videojs-contrib-quality-levels"

import forward_icon from "@/assets/forward_icon.svg"
import skip_icon from "@/assets/skip_icon.svg"
import rewind_icon from "@/assets/rewind_icon.svg"
import moviePath from "@/assets/first_to_the_gate.mp4"
// import moviePath from "@/assets/first_to_the_gate.mkv"

export default {
  name: "VideoPlayer",
  props: { options: Object },
  data() {
    return {
      icons: {
        forward_icon,
        skip_icon,
        rewind_icon,
      },
      player: null,
      videoOptions: {
        fluid: true,
        autoplay: true,
        controls: true,
        closeButton: true,
        playbackRates: [0.5, 1, 1.5, 2],
        titleBar: true,

        controlBar: {
          currentTimeDisplay: true,
          remainingTimeDisplay: false,
          timeDivider: true,
          durationDisplay: true,
          pictureInPictureToggle: false,
        },
        plugins: {
          hotkeys: {
            volumeStep: 0.1,
            seekStep: 10,
            enableModifiersForNumbers: false,
          },
          httpSourceSelector: {
            default: "auto",
          },
        },
        sources: [
          // {
          //   src: moviePath,
          //   type: "video/mp4",
          // },
          {
            src: moviePath,
            type: "video/x-matroska",
          },
          // {
          //   src: "https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8",
          //   type: "application/x-mpegURL",
          // },
          // {
          //   src: "https://dash.akamaized.net/dash264/TestCasesHD/2b/qualcomm/1/MultiResMPEG2.mpd",
          //   type: "application/dash+xml",
          // },
        ],
        ...this.options,
      },
    }
  },
  mounted() {
    // console.log(moviePath)
    const player = videojs(this.$refs.videoPlayer, this.videoOptions, () => {
      this.player.log("onPlayerReady", this)

      const rewindBtn = player.controlBar.addChild("button", {}, 2)
      rewindBtn.handleClick = () => this.rewindVideoOn(-10)
      const rewindBtnDom = rewindBtn.el()

      const forwardBtn = player.controlBar.addChild("button", {}, 3)
      forwardBtn.handleClick = () => this.rewindVideoOn(10)
      const forwardDom = forwardBtn.el()

      rewindBtnDom.innerHTML = `<img src="${rewind_icon}"></img>`
      forwardDom.innerHTML = `<img src="${forward_icon}"></img>`
    })

    player.getChild("CloseButton").handleClick = () => this.onCloseVideo()

    window.pr = player
    window.videojs = videojs

    this.player = player
  },
  beforeUnmount() {
    if (this.player) {
      this.player.dispose()
    }
  },
  methods: {
    rewindVideoOn(sec) {
      const newValue = this.player.currentTime() + sec
      this.player.currentTime(newValue)
    },
    onCloseVideo() {
      this.$store.state.movieIsPlaying = false
    },
  },
}
</script>
