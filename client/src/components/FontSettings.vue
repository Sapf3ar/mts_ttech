<template>
  <div class="dropdown-menu p-3" @click.stop>
    <div>
      <p class="text-muted mb-1">Шрифт</p>
      <div>
        <input
          v-model="font"
          value="Montserrat"
          class="form-check-input"
          type="radio"
          name="font"
          id="defaultFont"
        />
        <label class="form-check-label ms-2" for="defaultFont">
          Montserrat
        </label>
      </div>
      <div>
        <input
          v-model="font"
          value="PT Serif"
          class="form-check-input"
          type="radio"
          name="font"
          id="VIFont"
        />
        <label class="form-check-label ms-2" for="VIFont"> PT Serif </label>
      </div>
    </div>
    <div class="mt-3">
      <p class="text-muted mb-1 text-nowrap">Межбуквенный интервал</p>
      <div>
        <input
          v-model="letterSpacing"
          class="form-check-input"
          type="radio"
          name="letter-spacing"
          value="0px"
          id="defaultLS"
        />
        <label class="form-check-label ms-2" for="defaultLS">
          Стандартный (0 пт)</label
        >
      </div>
      <div>
        <input
          v-model="letterSpacing"
          class="form-check-input"
          type="radio"
          name="letter-spacing"
          value="0.5px"
          id="mdLS"
        />
        <label class="form-check-label ms-2" for="mdLS">
          Средний (0.5 пт)</label
        >
      </div>
      <div>
        <input
          v-model="letterSpacing"
          class="form-check-input"
          type="radio"
          name="letter-spacing"
          value="1px"
          id="lgLS"
        />
        <label class="form-check-label ms-2" for="lgLS"> Большой (1 пт)</label>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {}
  },
  computed: {
    font: {
      get() {
        return this.$store.state.font
      },
      set(value) {
        this.$store.dispatch("setFont", value)
      },
    },
    letterSpacing: {
      get() {
        return this.$store.state.letterSpacing
      },
      set(value) {
        this.$store.dispatch("setLetterSpacing", value)
      },
    },
  },
}
</script>
