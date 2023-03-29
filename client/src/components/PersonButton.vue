<template>
  <img :src="defaultLogo" alt="person_logo" />
</template>

<script>
import defaultLogo from "@/assets/icon_person.png"
export default {
  name: "PersonButton",
  data() {
    return { defaultLogo }
  },
}
</script>
