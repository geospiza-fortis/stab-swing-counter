import { writable } from "svelte/store";

export let currentTime = writable(0);
export let paused = writable(true);
