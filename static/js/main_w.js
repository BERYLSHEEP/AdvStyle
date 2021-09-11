function connectImageAndSlider(img, slider, attribute) {
        slider.oninput = (evt) => {
          img_id = evt.target.value;
          img.src = 'static/results/'+ attribute +'/'+ String(img_id) + '.jpg?ran=' + Math.random();
          }
        }
contentImg = document.getElementById('content-img');
contentImg.onerror = () => {
    alert("Error loading " + contentImg.src + ".");
}
attrDir = '';

contentImgSlider = document.getElementById('edit-ratio');
select_A = document.getElementById("hair-color-select");
index_A = select_A.selectedIndex;
attrDir += String(index_A);

select_B = document.getElementById("open-mouth-select");
index_B = select_B.selectedIndex;
attrDir += String(index_B);

select_C = document.getElementById("comic-style-select");
index_C = select_C.selectedIndex;
attrDir += String(index_C);

select_D = document.getElementById("FFHQ-style-select");
index_D = select_D.selectedIndex;
attrDir += String(index_D);

select_E = document.getElementById("itomugi-kun-style-select");
index_E = select_E.selectedIndex;
attrDir += String(index_E);

select_F = document.getElementById("chibi-maruko-style-select");
index_F = select_F.selectedIndex;
attrDir += String(index_F);

select_G = document.getElementById("onepiece-style-select");
index_G = select_G.selectedIndex;
attrDir += String(index_G);
attrDir += 'w'
contentImg.src = 'static/results/'+ attrDir +'/50.jpg?ran=' + Math.random();
connectImageAndSlider(contentImg, contentImgSlider, attrDir);

styleButton = document.getElementById("style-button")
function disableStylizeButtons() {
    styleButton.disabled = true;
//    styleButton.textContent = 'Editing...';
}

styleButton.disabled = false;
//styleButton.textContent = 'Edit';
function loadImage(event, imgElement) {
  const reader = new FileReader();
  reader.onload = (e) => {
    imgElement.src = e.target.result;
  };
  reader.readAsDataURL(event.target.files[0]);
}

function loadContent(event) {
  loadImage(event, contentImg);
}
