import{n as d}from"./vue.4dfaac09.js";const e={start:()=>{const n=document.body,t=document.createElement("div");t.setAttribute("class","loading-next"),t.innerHTML=`
			<div class="loading-next-box">
				<div class="loading-next-box-warp">
					<div class="loading-next-box-item"></div>
					<div class="loading-next-box-item"></div>
					<div class="loading-next-box-item"></div>
					<div class="loading-next-box-item"></div>
					<div class="loading-next-box-item"></div>
					<div class="loading-next-box-item"></div>
					<div class="loading-next-box-item"></div>
					<div class="loading-next-box-item"></div>
					<div class="loading-next-box-item"></div>
				</div>
			</div>
		`,n.insertBefore(t,n.childNodes[0]),window.nextLoading=!0},done:(n=0)=>{d(()=>{setTimeout(()=>{var i;window.nextLoading=!1;const t=document.querySelector(".loading-next");(i=t==null?void 0:t.parentNode)==null||i.removeChild(t)},n)})}};export{e as N};
