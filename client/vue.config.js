const { defineConfig } = require("@vue/cli-service")
const path = require("path")

module.exports = defineConfig({
  chainWebpack: (config) => {
    const types = ["vue-modules", "vue", "normal-modules", "normal"]
    types.forEach((type) =>
      addStyleResource(config.module.rule("stylus").oneOf(type))
    )
  },

  transpileDependencies: true,
  filenameHashing: false,
})

module.exports = {
  devServer: {
    allowedHosts: "all",
  },
  filenameHashing: false,
}

function addStyleResource(rule) {
  rule
    .use("style-resource")
    .loader("style-resources-loader")
    .options({
      patterns: [path.resolve(__dirname, "./src/styles/imports.styl")],
    })
}
