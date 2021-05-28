console.log('hi')
var startdate=d3.select('#startdate')
var enddate=d3.select('#enddate')

var filter_btn=d3.select('#filter-btn')

filter_btn.on('click', function(e){
	e.preventDefault()
	console.log(startdate.property('value'))
	console.log(enddate.property('value'))
})

// / //Step 1:
// // from data.js
// var tableData = data;
// console.log(data)

// // YOUR CODE HERE!

// // Step 2:
// //Get a reference to the table body
// var tbody = d3.select('tbody');
// //console.log the weather data 
// console.log(tableData)

// //Step 3: 
// //Loop through data and console.log each ufoSightings object
// tableData.forEach(function(ufoSightings) {
// 	console.log(ufoSightings);
// });

// // Step 4:

// //Use d3 to append one table row 'tr' for each ufoSightings object
// // tableData.forEach(function(ufoSightings) {
// // 	console.log(ufoSightings);
// // 	var row= tbody.append("tr");
// // });

// //Use Object.entries to console.log each ufoSightings value
// tableData.forEach(function(ufoSightings) {
// 	console.log(ufoSightings);
// 	var row =tbody.append("tr");

// //k is short for key & v is short for value 
// Object.entries(ufoSightings).forEach(function([k,v]) {
// 	console.log(k,v);
// 	});
// });

// // Step 5:
// //Use d3 to append 1 cell per ufoSightings value
// tableData.forEach(function(ufoSightings) {
// 	console.log(ufoSightings);
// 	var row =tbody.append("tr");
// 	Object.entries(ufoSightings).forEach(function([k,v]) {
// 	console.log(k,v);
// 	var cell=row.append("td");
// 	cell.text(v);
// 	});
// });

// //Step 6:
// //Select the button and the form
// var button = d3.select("#filter-btn");

// var form =d3.select("#datetime");

// //Step 7:
// //Create event handlers
// button.on("click", runEnter);

// form.on("submit", runEnter);

// //Step 8:
// //Complete the event handeler function for the form
// function runEnter() {
// 	// Prevent the page from refreshing
// 	d3.event.preventDefault();

// 	//Select the input element and get the raw html node
// 	var ufoElement =d3.select("#datetime");
// 	//get the value property of the ie (input element)
// 	var valueProp =ufoElement.property("value");

// 	// console.log(valueProp);
// 	// console.log(tableData);

// var filteredData= tableData.filter(ufoDate=> ufoDate.datetime === valueProp);
// // console.log(filteredData);

// //Step 9:
// //Then select the unordered list element by class name
// var table= d3.select('tbody');

// //Step 10: 
// // remove any children from the list to
// table.html("");

// filteredData.forEach((ufoSightings)=> {
// 	console.log(ufoSightings);
// 	var row =tbody.append("tr");
// 	Object.entries(ufoSightings).forEach(([k,v])=> {
// 	var cell=row.append("td");
// 	cell.text(v);
// 	});
// });


// };

