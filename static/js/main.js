function connectImageAndSlider(img, slider, attribute) {
        slider.oninput = (evt) => {
          img_id = evt.target.value;
          img.src = 'static/results/'+ String(img_id) + '.jpg?ran=' + Math.random();
          }
        }
contentImg = document.getElementById('content-img');
contentImg.onerror = () => {
    alert("Error loading " + contentImg.src + ".");
}
contentImgSlider1 = document.getElementById('step1');
contentImgSlider2 = document.getElementById('step2');
contentImgSlider3 = document.getElementById('step3');
contentImgSlider4 = document.getElementById('step4');
contentImgSlider5 = document.getElementById('step5');
contentImgSlider6 = document.getElementById('step6');
contentImgSlider7 = document.getElementById('step7');
contentImgSlider8 = document.getElementById('step8');
contentImgSlider9 = document.getElementById('step9');
contentImgSlider0 = document.getElementById('step0');
contentImg.src = 'static/results/0.jpg?ran=' + Math.random();
connectImageAndSlider(contentImg, contentImgSlider1);
connectImageAndSlider(contentImg, contentImgSlider2);
connectImageAndSlider(contentImg, contentImgSlider3);
connectImageAndSlider(contentImg, contentImgSlider4);
connectImageAndSlider(contentImg, contentImgSlider5);
connectImageAndSlider(contentImg, contentImgSlider6);
connectImageAndSlider(contentImg, contentImgSlider7);
connectImageAndSlider(contentImg, contentImgSlider8);
connectImageAndSlider(contentImg, contentImgSlider9);
connectImageAndSlider(contentImg, contentImgSlider0);

Button1 = document.getElementById("button1")
Button2 = document.getElementById("button2")
Button3 = document.getElementById("button3")
Button4 = document.getElementById("button4")
Button5 = document.getElementById("button5")
Button6 = document.getElementById("button6")
Button7 = document.getElementById("button7")
Button8 = document.getElementById("button8")
Button9 = document.getElementById("button9")
Button0 = document.getElementById("button0")
function disableAllButtons() {
    Button1.disabled = true;
    Button2.disabled = true;
    Button3.disabled = true;
    Button4.disabled = true;
    Button5.disabled = true;
    Button6.disabled = true;
    Button7.disabled = true;
    Button8.disabled = true;
    Button9.disabled = true;
    Button0.disabled = true;
//    styleButton.textContent = 'Editing...';
}

Button1.disabled = false;
Button2.disabled = false;
Button3.disabled = false;
Button4.disabled = false;
Button5.disabled = false;
Button6.disabled = false;
Button7.disabled = false;
Button8.disabled = false;
Button9.disabled = false;
Button0.disabled = false;
//styleButton.textContent = 'Edit';


