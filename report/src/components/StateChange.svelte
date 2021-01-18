<script>
  import { zip } from "lodash";
  import Papa from "papaparse";
  import { onMount } from "svelte";

  let data;
  let labelData;
  let keys = ["pos", "label", "start", "end", "duration", "stab_pct"];

  function transform(data) {
    // What frame did it start, and what frame did it end.
    let res = [];
    let start = 0;
    let value = "other";
    for (let i = 0; i < data.length; i++) {
      let e = data[i];
      if (e != value) {
        if (value != "other") {
          res.push({
            pos: res.length,
            label: value,
            start: start,
            end: i,
            // time since the last attack
            duration: res.length > 0 ? start - res[res.length - 1].start : 0,
            stab_pct: (
              (res.filter(x => x.label == "stab").length + (value == "stab")) /
              (res.length + 1)
            ).toFixed(2)
          });
        }
        start = i;
        value = e;
      }
    }
    return res;
  }

  onMount(async () => {
    let resp = await fetch("pred.json");
    data = await resp.json();
    labelData = transform(data);
  });
</script>

<style>
  table,
  th,
  td {
    border: 1px solid;
  }
</style>

{#if labelData}
  <table>
    <thead>
      {#each keys as key}
        <th>{key}</th>
      {/each}
    </thead>
    <tbody>
      {#each labelData as row}
        <tr>
          {#each keys as key}
            <td>{row[key]}</td>
          {/each}
        </tr>
      {/each}
    </tbody>
  </table>
{/if}
