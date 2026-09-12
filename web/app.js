const form=document.querySelector("#askForm"),input=document.querySelector("#queryInput"),submitButton=document.querySelector("#submitButton"),loading=document.querySelector("#loading"),loadingText=document.querySelector("#loadingText"),result=document.querySelector("#result"),errorPanel=document.querySelector("#errorPanel"),errorText=document.querySelector("#errorText"),agentCards=[...document.querySelectorAll(".process article")];
const statusClass={answered:"",answered_partial:"warning",insufficient:"warning",needs_clarification:"info",rejected:"danger",error:"danger"};
const stageCopy=["Agent 1 is checking scope and planning retrieval…","Agent 2 is searching dense and sparse indexes…","Agent 3 is verifying the supporting passages…","Agent 4 is composing a grounded answer…"];
function setAgentStage(index){agentCards.forEach((card,i)=>{card.classList.toggle("active",i===index);card.classList.toggle("complete",i<index)});loadingText.textContent=stageCopy[index]}
function textWithCitations(text){const fragment=document.createDocumentFragment();text.split(/(\[\d+\])/g).forEach(part=>{const match=part.match(/^\[(\d+)\]$/);if(!match){fragment.append(document.createTextNode(part));return}const button=document.createElement("button");button.className="cite-marker";button.type="button";button.textContent=part;button.setAttribute("aria-label",`Open source ${match[1]}`);button.addEventListener("click",()=>focusCitation(match[1]));fragment.append(button)});return fragment}
function renderAnswer(text){const root=document.querySelector("#answerCopy");root.replaceChildren();let list=null;text.split("\n").forEach(line=>{const item=line.match(/^\s*(?:[-*]|\d+\.)\s+(.+)/);if(item){if(!list){list=document.createElement(/^\s*\d+\./.test(line)?"ol":"ul");root.append(list)}const li=document.createElement("li");li.append(textWithCitations(item[1]));list.append(li);return}list=null;if(!line.trim())return;const paragraph=document.createElement("p");paragraph.append(textWithCitations(line));root.append(paragraph)})}
function focusCitation(marker){const card=document.querySelector(`#citation-${marker}`);if(!card)return;card.scrollIntoView({behavior:"smooth",block:"center"});card.classList.add("flash");window.setTimeout(()=>card.classList.remove("flash"),1200)}
function getTableData(citation){
  const rows=Array.isArray(citation.table_rows)?citation.table_rows.filter(row=>row&&typeof row==="object"):[];
  const declared=Array.isArray(citation.table_headers)?citation.table_headers.filter(header=>typeof header==="string"&&header.trim()):[];
  const discovered=rows.flatMap(row=>Object.keys(row));
  const headers=[...new Set([...declared,...discovered])];
  return {headers,rows};
}
function renderCitationTable(citation){
  const {headers,rows}=getTableData(citation);
  if(!headers.length||!rows.length)return null;

  const region=document.createElement("div");
  region.className="citation-table-wrap";
  region.tabIndex=0;
  region.setAttribute("role","region");
  region.setAttribute("aria-label","Table evidence: "+(citation.table_caption||citation.section||citation.source));

  const table=document.createElement("table");
  table.className="citation-table";
  if(citation.table_caption){
    const caption=document.createElement("caption");
    caption.textContent=citation.table_caption;
    table.append(caption);
  }

  const thead=document.createElement("thead"),headerRow=document.createElement("tr");
  headers.forEach(header=>{
    const cell=document.createElement("th");
    cell.scope="col";
    cell.textContent=header;
    headerRow.append(cell);
  });
  thead.append(headerRow);

  const tbody=document.createElement("tbody");
  rows.forEach(row=>{
    const tableRow=document.createElement("tr");
    headers.forEach(header=>{
      const cell=document.createElement("td");
      const value=row[header];
      cell.textContent=value==null?"":String(value);
      tableRow.append(cell);
    });
    tbody.append(tableRow);
  });
  table.append(thead,tbody);
  region.append(table);
  return region;
}
function renderCitations(citations){
  const root=document.querySelector("#citations"),section=document.querySelector("#evidenceSection");
  root.replaceChildren();
  section.hidden=citations.length===0;
  document.querySelector("#citationCount").textContent=citations.length+" PASSAGE"+(citations.length===1?"":"S");

  citations.forEach(citation=>{
    const tableContent=citation.chunk_type==="table"?renderCitationTable(citation):null;
    const card=document.createElement("article");
    card.className="citation"+(tableContent?" citation--table":"");
    card.id="citation-"+citation.marker;

    const head=document.createElement("div");
    head.className="citation-head";
    const marker=document.createElement("span");
    marker.className="citation-number";
    marker.textContent=String(citation.marker).padStart(2,"0");
    const source=document.createElement("span");
    source.className="citation-source";
    source.textContent=citation.source;
    const pages=document.createElement("span");
    pages.className="citation-page";
    pages.textContent=citation.pages;
    head.append(marker,source,pages);

    const sectionName=document.createElement("p");
    sectionName.className="citation-section";
    sectionName.textContent=citation.section||"General guidance";
    card.append(head,sectionName);

    if(tableContent){
      const tableMeta=document.createElement("div");
      tableMeta.className="citation-table-meta";
      const type=document.createElement("span");
      type.className="citation-type";
      type.textContent="TABLE";
      tableMeta.append(type);
      if(citation.table_row_start){
        const rowRange=document.createElement("span");
        const end=citation.table_row_end||citation.table_row_start;
        rowRange.textContent=citation.table_row_start===end?"ROW "+end:"ROWS "+citation.table_row_start+"–"+end;
        tableMeta.append(rowRange);
      }
      card.append(tableMeta,tableContent);
    }else{
      const quote=document.createElement("blockquote");
      quote.textContent="“"+citation.quote+"”";
      card.append(quote);
    }
    root.append(card);
  });
}
function renderTrace(trace,rounds){const root=document.querySelector("#traceList");root.replaceChildren();document.querySelector("#traceSummary").textContent=`${trace.length} EVENTS · ${rounds||0} ROUND${rounds===1?"":"S"}`;trace.forEach(event=>{const row=document.createElement("div");row.className="trace-item";const round=document.createElement("span");round.className="round";round.textContent=`R${event.round||0}`;const node=document.createElement("span");node.className="node";node.textContent=event.node;const detail=document.createElement("span");detail.className="detail";detail.textContent=event.detail;row.append(round,node,detail);root.append(row)});document.querySelector("#tracePanel").hidden=trace.length===0}
function renderResult(data){const payload=data.answer,status=data.status||"error",statusPill=document.querySelector("#statusPill");statusPill.textContent=status.replaceAll("_"," ");statusPill.className=`status-pill ${statusClass[status]||""}`;document.querySelector("#confidence").textContent=payload?.confidence?`${payload.confidence} confidence`:"";renderAnswer(data.response||"No response was returned.");renderCitations(payload?.citations||[]);renderTrace(data.trace||[],data.rounds);const caveat=document.querySelector("#caveat");caveat.textContent=payload?.caveats||"";caveat.hidden=!payload?.caveats;result.hidden=false;result.scrollIntoView({behavior:"smooth",block:"start"})}
function showError(message){errorText.textContent=message;errorPanel.hidden=false;errorPanel.scrollIntoView({behavior:"smooth",block:"center"})}
async function ask(query){result.hidden=true;errorPanel.hidden=true;loading.hidden=false;submitButton.disabled=true;let stage=0;setAgentStage(stage);const ticker=window.setInterval(()=>{stage=Math.min(stage+1,stageCopy.length-1);setAgentStage(stage)},2600);try{const response=await fetch("/api/ask",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({query})});const data=await response.json().catch(()=>({}));if(!response.ok)throw new Error(data.error||`Request failed (${response.status})`);agentCards.forEach(card=>{card.classList.remove("active");card.classList.add("complete")});renderResult(data)}catch(error){agentCards.forEach(card=>card.classList.remove("active","complete"));showError(error.message||"The pipeline could not complete this request.")}finally{window.clearInterval(ticker);loading.hidden=true;submitButton.disabled=false}}
form.addEventListener("submit",event=>{event.preventDefault();const query=input.value.trim();if(query)ask(query)});document.querySelectorAll("[data-query]").forEach(button=>button.addEventListener("click",()=>{input.value=button.dataset.query;input.focus()}));document.querySelector("#dismissError").addEventListener("click",()=>{errorPanel.hidden=true});
fetch("/api/status").then(async response=>{const data=await response.json();if(!response.ok)throw new Error(data.error||"Corpus unavailable");const state=document.querySelector("#systemState");state.classList.add(data.ready?"ready":"error");document.querySelector("#systemText").textContent=data.ready?`${data.indexed_chunks} passages · ${data.documents} sources`:"Index is empty — run ingest"}).catch(error=>{document.querySelector("#systemState").classList.add("error");document.querySelector("#systemText").textContent=error.message});
