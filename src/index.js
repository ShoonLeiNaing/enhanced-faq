const express = require("express");
const app = express();
const cors = require("cors");
const bodyParser = require("body-parser");
const { submitQuestion, storeData } = require("./services/geminiServices");
require("dotenv").config();
const port = 4000;
app.use(cors({ origin: "*" }));

app.use(
  bodyParser.urlencoded({
    // to support URL-encoded bodies
    extended: true,
  })
);
app.use(bodyParser.json());

app.use(express.json());

app.post("/api/submit-question", submitQuestion);

app.listen(port, async () => {
  // await storeData();
  console.log(`Listening at http://0.0.0.0:${port}`);
});
