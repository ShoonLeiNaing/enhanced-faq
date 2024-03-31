const { GoogleGenerativeAI } = require("@google/generative-ai");
const {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings,
} = require("@langchain/google-genai");
const {
  ChatPromptTemplate,
  SystemMessagePromptTemplate,
  HumanMessagePromptTemplate,
} = require("@langchain/core/prompts");
const {
  EmbeddingsFilter,
} = require("langchain/retrievers/document_compressors/embeddings_filter");
const {
  ContextualCompressionRetriever,
} = require("langchain/retrievers/contextual_compression");
const { ConversationalRetrievalQAChain } = require("langchain/chains");
const { BufferWindowMemory } = require("langchain/memory");
const { getRetriever } = require("./vectorStore");
const { useDirectoryLoader } = require("./fileloader");

let compression_retriever, llm;

const storeData = async () => {
  /* Create Training Data for Chatbot */
  // const documents = await usePuppeteer(urls);
  const documents = await useDirectoryLoader("public/data");

  const embeddings = new GoogleGenerativeAIEmbeddings({
    modelName: "embedding-001",
  });

  const collectionName = "RIC-enhanced-FAQ";
  const retriever = await getRetriever(documents, embeddings, collectionName);
  llm = new ChatGoogleGenerativeAI({ modelName: "gemini-pro" });

  /* Creating Compression Retriever for Accurate Results */
  const embeddings_filter = new EmbeddingsFilter({
    embeddings,
    similarityThreshold: 0.7,
    k: 10,
  });

  compression_retriever = new ContextualCompressionRetriever({
    baseCompressor: embeddings_filter,
    baseRetriever: retriever,
  });
};

const submitQuestion = async (req, res) => {
  /* Creating Prompt */
  // const system_template = `
  // When user ask with other language such as Chinese, Japanese and Indonesia, then you translate first into English and Answer the question as detailed as possible from the provided context. Use the following pieces of context to answer the users question.
  // If you don't know the answer, just say that you don't know, don't try to make up an answer.
  // ----------------
  // {context}`;
  // const messages = [
  //   SystemMessagePromptTemplate.fromTemplate(system_template),
  //   HumanMessagePromptTemplate.fromTemplate("{question}"),
  // ];
  // const prompt = ChatPromptTemplate.fromMessages(messages);
  // /* Creating Memory Instance */
  // const memory = new BufferWindowMemory({
  //   memoryKey: "chat_history",
  //   inputKey: "question",
  //   outputKey: "text",
  //   k: 3,
  //   returnMessages: true,
  // });
  // /* Creating Question Chain */
  // const chain = ConversationalRetrievalQAChain.fromLLM(
  //   llm,
  //   compression_retriever,
  //   {
  //     returnSourceDocuments: true,
  //     memory: memory,
  //     verbose: true,
  //     qaChainOptions: {
  //       type: "stuff",
  //       prompt: prompt,
  //     },
  //   }
  // );
  // try {
  //   /* Invoking Chain for Q&A */
  //   const result = await chain.invoke({
  //     question: req.body.question,
  //     chat_history: memory,
  //   });
  //   const answer = await result.text;
  //   const sources = await result.sourceDocuments;
  //   res.status(200).json({ msg: answer });
  // } catch (error) {
  //   console.error(error);
  //   res.status(400).json({ msg: "Something went wrong" });
  // }
  try {
    const genAI = new GoogleGenerativeAI(
      "AIzaSyCBFARQS8sOgOK0Klc__Gtdsei8plZpWfQ"
    );
    const model = genAI.getGenerativeModel({ model: "gemini-pro" });
    const chat = model.startChat({
      history: [
        {
          role: "user",
          parts:
            "how many options for internship at ICT Program Rangsit International college?",
        },
        {
          role: "model",
          parts:
            "There are two options for Internship at RIC ICT Program (Option 1) Internship and Senior Project (6 credits) and (Option 2) ICT 498 Co-operative Education (6 credits)",
        },
        {
          role: "user",
          parts: "Tell me details of internship options.",
        },
        {
          role: "model",
          parts:
            "There are two options for Internship at RIC ICT Program (Option 1) Internship and Senior Project (6 credits) 1(0-35-18) ICT 493 Internship (1 credit) Students will enter an industrial training program with the industrial sector, the business sector, or other public organizations for a period of at least 280 hours. Students need to work at least 280 hours or 2 months for internship. Student is required to do the report. 2(2-0-4) ICT 494 Introduction to Senior Project (2 credits) Before you start the internship or co operative education, Students must earned at least 45 credits of required major courses or acquire consent from Head of Program. Creation of senior project topics; outlining and drafting senior projects; citations and reference formats; writing introduction for senior projects; compilation of theories; frameworks and existed works and/or related researches; writing for senior project methodologies; development and presentation of models and/or tools used in senior projects; preliminary examination of senior project proposals. 3(0-6-3) ICT 495 Senior Project (3 credits) . Students must finished ICT 494 Introduction to Senior Project and/or must earned at least 50 credits of required major courses or acquire consent of Head of Program. Completion of senior projects; printing standard for full senior projects; preparation for final examination of senior projects; final examination of senior projects. 6(0-35-18). (Option 2) ICT 498 Co-operative Education (6 credits) Students must earned at least 60 credits of major courses or acquire consent from Head of Program. Minimum of 16 weeks or 560 hours must be achieved to complete co operative education. This program is designed to enhance traditional academic study by providing students the opportunity to integrate academic study and classroom theory with productive and real-life work experience in Information and Communication Technology field at a work site in the community. It also provides the time to explore occupational areas and to develop skills, knowledge and attitudes needed to become productive and satisfied members of society.",
        },
        {
          role: "user",
          parts: "Timeline for internship and co-operative",
        },
        {
          role: "model",
          parts:
            "There are 12 steps for Timeline for internship and co-operative, each step has completion percentages:\n1. Looking for work placement (ICTCE01 Co-Operative Education Application Form) - 0% complete.\n2. Work Placement Accept - 5% complete.\n3. Job Description form sign-up from workplace (ICTCE02 Co-Operative Education Job Description form) - 10% complete.\n4. ICT Department issue the letter for work placement (Letter of request for work placement * giving to the employer) - 10% complete.\n5. Teacher visiting (ICTCE03 confirmation form of workplace supervision of Coop Edu supervision program) - 15% complete.\n6. Weekly progress report (ICTCE04 Weekly process report form) - 35% complete.\n7. Evaluate work performance (ICTCE05 Co-operative education evaluation form from job supervisor) - 50% complete.\n8. Evaluate internship report (ICTCE06 Student evaluation form from job advisor/supervisor) - 60% complete.\n9. Consent to publish report (ICTCE07 – Letter of consent to publish report/co-operative education project) - 70% complete.\n10. Submit Report (cooperative education report for 1 job supervisor, 2 co-operative supervisors, 3 the committee) - 80% complete.\n11. Presentation (Student’s evaluation form (for job advisor/supervisor for 1 job supervisor, 2 co-operative supervisors, 3 the committee) - 90% complete.\n12. Amendment & Submit final draft report (within 2 weeks after presentation) - 100% complete.",
        },
        {
          role: "user",
          parts:
            "Tell me steps of GRADUATION REQUEST PROCESS RANGSIT UNIVERSITY",
        },
        {
          role: "model",
          parts:
            "Keeping in mind to note the specific period for near-graduated students to request graduation according to the academic calendar, the steps are:\nStart Graduation Request (0%).\nVisit the Registrar’s office website or directly access http://registrar.rsu.ac.th/applyforgraduate (period for near-graduated).\nAuthentication (10%).\nLog in with your Username (with 'u' in front) and Password (same as the Intranet system).\nInput Information (20%).\nFill in your address details for document dispatch.\nPrint Forms (30%).\nPrint out the Curriculum Structure Form (RSU.36) and Graduation Request Form (RSU.62).\nPayment (40%).\nPay the 200 Baht fee via the provided payment channels (QR code available).\nPhoto Submission (50%).\nObtain a 2-inch photo in graduation gown against a blue background.\nAdvisor Approval (60%).\nGet the Curriculum Structure Form (RSU.36) approved and signed by your advisor.\nForm Submission (70%).\nSubmit the RSU.36 and RSU.62 forms along with the photograph to the Registrar’s office before the deadline (period for near-graduated).\nConfirmation (80%).\nReceive acknowledgment of submission from the Registrar’s office.\nFinal Verification (90%).\nWait for the verification process by the University Council.\nGraduation Approval (100%).\nReceive confirmation of graduation approval.",
        },
        {
          role: "user",
          parts:
            "How many forms do I need to fill for internship and co-operative education?",
        },
        {
          role: "model",
          parts:
            "There are 7 application forms for internship and co-operative education:\n1. ICTCE01 - Looking for work placement (Co-operative Education Application Form)\n2. ICTCE02 - Job description form sign-up from workplace (Co-operative Education Job Description Form)\n3. ICTCE03 - (Confirmation Form of Workplace Supervision of Co-operative Education Supervision Program)\n4. ICTCE04 - Weekly Progress Report (Weekly Progress Report Form)\n5. ICTCE05 - Evaluate Work Performance Co-operative Education Evaluation Form (For Job Supervisor)\n6. ICTCE06 - Evaluate Internship Report Student’s Evaluation Form (For Job Advisor/Supervisor)\n7. ICTCE07 - Consent to Publish Report (Letter of Consent to Publish Report/Co-operative Education Project)",
        },
        {
          role: "user",
          parts: "How many internship options are there in RIC?",
        },
        {
          role: "model",
          parts:
            "There are two internship options: 1. Internship and 2. Co-operative Education.",
        },
        {
          role: "user",
          parts: "How to find internship or co-operative education?",
        },
        {
          role: "model",
          parts:
            "You can try to find internships from these platforms: glints, internjobs.com, experience by simplicity, internsg, glassdoor, hiredly, LinkedIn, wayup, Asia Internship program, and gradsingapore.",
        },
        {
          role: "user",
          parts: "What are the admissions requirements?",
        },
        {
          role: "model",
          parts:
            "The academic entry requirement is a high school diploma or equivalent and IELTS 5 or equivalent.",
        },
        {
          role: "user",
          parts: "What if I don’t have any recent IELTS scores?",
        },
        {
          role: "model",
          parts:
            "Students can take an English Placement Test (free-of-charge) once they arrive at the university. If a student’s English level is below the minimum requirement of IELTS 5, the student would need to take some English subjects before starting the program. Please see the section English Placement Test for more details.",
        },
        {
          role: "user",
          parts: "How long does the admissions process take?",
        },
        {
          role: "model",
          parts:
            "Once we have the completed application file, including all required materials, it usually takes a few days to a week for application processing, review, and for your Letter of Acceptance to be sent by email. You will then be required to pay a non-refundable partial payment 30% of tuition fees of the semester before your visa supporting document is granted. The remaining 70% of the fees are paid once you arrived at the university. Please note, the partial payment may be changed based on your country of origin.",
        },
        {
          role: "user",
          parts: "Where do I get my visa and how long does it take to have it?",
        },
        {
          role: "user",
          parts:
            "Tell me steps of GRADUATION REQUEST PROCESS RANGSIT UNIVERSITY",
        },
        {
          role: "model",
          parts:
            "Keeping in mind to note the specific period for near-graduated students to request graduation according to the academic calendar, the steps are:\nStart Graduation Request (0%).\nVisit the Registrar’s office website or directly access http://registrar.rsu.ac.th/applyforgraduate (period for near-graduated).\nAuthentication (10%).\nLog in with your Username (with 'u' in front) and Password (same as the Intranet system).\nInput Information (20%).\nFill in your address details for document dispatch.\nPrint Forms (30%).\nPrint out the Curriculum Structure Form (RSU.36) and Graduation Request Form (RSU.62).\nPayment (40%).\nPay the 200 Baht fee via the provided payment channels (QR code available).\nPhoto Submission (50%).\nObtain a 2-inch photo in graduation gown against a blue background.\nAdvisor Approval (60%).\nGet the Curriculum Structure Form (RSU.36) approved and signed by your advisor.\nForm Submission (70%).\nSubmit the RSU.36 and RSU.62 forms along with the photograph to the Registrar’s office before the deadline (period for near-graduated).\nConfirmation (80%).\nReceive acknowledgment of submission from the Registrar’s office.\nFinal Verification (90%).\nWait for the verification process by the University Council.\nGraduation Approval (100%).\nReceive confirmation of graduation approval.",
        },
        {
          role: "user",
          parts:
            "How many forms do I need to fill for internship and co-operative education?",
        },
        {
          role: "model",
          parts:
            "There are 7 application forms for internship and co-operative education:\n1. ICTCE01 - Looking for work placement (Co-operative Education Application Form)\n2. ICTCE02 - Job description form sign up from workplace (Co-operative Education Job Description Form)\n3. ICTCE03 - (Confirmation Form of Workplace Supervision of Co-operative Education Supervision Program)\n4. ICTCE04 - Weekly Progress Report (Weekly Progress Report Form)\n5. ICTCE05 - Evaluate Work Performance Co-operative Education Evaluation Form (For Job Supervisor)\n6. ICTCE06 - Evaluate Internship Report Student’s Evaluation Form (For Job Advisor/Supervisor)\n7. ICTCE07 - Consent to Publish Report (Letter of Consent to Publish Report/Co-operative Education Project)",
        },
        {
          role: "user",
          parts: "How many internship options are there in RIC?",
        },
        {
          role: "model",
          parts:
            "There are two internship options: 1. Internship and 2. Co-operative Education.",
        },
        {
          role: "user",
          parts: "How to find internship or co-operative education?",
        },
        {
          role: "model",
          parts:
            "You can try to find internships from these platforms: glints, internjobs.com, experience by simplicity, internsg, glassdoor, hiredly, LinkedIn, wayup, Asia Internship program, and gradsingapore.",
        },
        {
          "role": "user",
          "parts": "What if you don’t have any recent IELTS scores?"
          },
          {
          "role": "model",
          "parts": "Students who don't have recent IELTS scores can typically take an English Placement Test, often provided free-of-charge, once they arrive at the university. If a student's English proficiency level is below the minimum requirement of IELTS 5, they may be required to take additional English language courses before beginning their program."
          },
          {
            "role": "user",
            "parts": "What are the admissions requirements?"
          },
          {
            "role": "model",
            "parts": "The academic entry requirement is a high school diploma or equivalent and IELTS 5 or equivalent. You can see more details in the English Requirements section."
          },
          {
            "role": "user",
            "parts": "How long does the admissions process take?"
          },
          {
            "role": "model",
            "parts": "Once the completed application file, including all required materials, is submitted, it typically takes a few days to a week for application processing, review, and for the Letter of Acceptance to be sent via email. Upon receipt of the Letter of Acceptance, you will be required to make a non-refundable partial payment of 30% of the tuition fees for the semester before your visa supporting document is granted. The remaining 70% of the fees are usually paid upon your arrival at the university. Please note that the partial payment may vary based on your country of origin."
          },
          {
            "role": "user",
            "parts": "Where do I get my visa and how long does it take to have it?"
          },
          {
            "role": "model",
            "parts": "Once the non-refundable partial payment of 30% of tuition fees is received, the university will process your visa supporting document and send it to you via email or postal mail. You can then lodge your visa application at the nearest Thai Embassy or consulate in your country. The processing time for the visa application can range from at least 1 week to 1 month, depending on the visa processing requirements of the Thai embassy or consulate in your country, which you have provided to the university during your online application."
          },
          {
            "role": "user",
            "parts": "What documents must I present to get my visa?"
          },
          {
            "role": "model",
            "parts": "To obtain your visa, you will need to present a scan copy of the visa issuance letter sent to your email, along with other required documents sent via EMS to the postal address you specified on the online application form. Once you receive the original visa issuance letter, all these documents must be submitted to the Thai embassy or consulate for your visa to be granted."
          },
          {
            "role": "user",
            "parts": "What if I studied before from another college or university?"
          },
          {
            "role": "model",
            "parts": "If you have previously completed units or subjects from another institution, whether in Thailand, overseas, or from a different degree at Rangsit University, you may apply for credit transfer. This could potentially reduce the time required to complete your current degree. The maximum allowable credit transfer is 75% of the total credits for the current program. Eligible subjects for credit transfer must have an exact equivalent credit value or higher, and a completed grade of C or above. For more details, please refer to the Credit Transfer section."
          },
          {
            "role": "user",
            "parts": "Do you require SAT/ACT scores?"
          },
          {
            "role": "model",
            "parts": "The university does not require SAT/ACT scores for admission to the international programs."
          },
          {
            "role": "user",
            "parts": "Do you require letters of reference?"
          },
          {
            "role": "model",
            "parts": "Letters of reference are not required, and they are not used as part of the admissions process."
          },
          {
            "role": "user",
            "parts": "How early should a student apply?"
          },
          {
            "role": "model",
            "parts": "It is advisable to apply at least 2 months before the semester starts. Applying early is important. The university offers intakes in Summer (June-July), 1st Semester (August to December), and 2nd Semester (January to May). Applications are accepted on a rolling basis and typically take at least 2 to 4 weeks until the student receives the visa."
          },
          {
            "role": "user",
            "parts": "What are your coursework requirements?"
          },
          {
            "role": "model",
            "parts": "For course requirements, please visit the Programs section of the website."
          },
          {
            "role": "user",
            "parts": "How much does it cost to attend the program?"
          },
          {
            "role": "model",
            "parts": "For current estimated tuition fees, please refer to the Tuition section of the website."
          },
          {
            "role": "user",
            "parts": "What is the next step once I receive my visa?"
          },
          {
            "role": "model",
            "parts": "Once you receive your visa, you can proceed to book your air ticket to fly to Thailand and start arranging your accommodation and airport pick-up through the university's website."
          },
          {
            "role": "user",
            "parts": "Are students required to live on campus?"
          },
          {
            "role": "model",
            "parts": "Students are not required to live on campus. Initially, students are accommodated in one of the off-campus apartments reserved by the International Service Center, which are conveniently located near or next to the university and easily accessible on foot."
          },
          {
            "role": "user",
            "parts": "Can I stay inside the campus dormitory?"
          },
          {
            "role": "model",
            "parts": "Due to the limited availability of on-campus accommodation, reservations must be made at least 2 months in advance. Students can initially stay at the off-campus dormitory provided, and once settled, they can request to reserve on-campus accommodation."
          },
          {
            "role": "user",
            "parts": "How do students apply for scholarships?"
          },
          {
            "role": "model",
            "parts": "To apply for scholarships, students must first complete the online scholarship application form for the relevant intake and academic year, and attach the required relevant documents. Upon receiving the application, the university will notify applicants of the schedule for a Skype interview, which will be at the specified local date and time."
          },
          {
            "role": "user",
            "parts": "How do you determine scholarship eligibility?"
          },
          {
            "role": "model",
            "parts": "Scholarship eligibility is determined by the scholarship committee through a review process that includes interviews. Factors considered for eligibility include, but are not limited to, grade point average (GPA), coursework, test scores, extracurricular activities, and community involvement."
          },          
      ],
    });
    const { question } = req.body;
    const result = await model.generateContent(question);
    const response = await result.response;
    const text = response.text();
    res.status(200).json({ msg: text });
  } catch (error) {
    console.error(error);
    res.status(400).json({ msg: "Something went wrong" });
  }
};

module.exports = {
  submitQuestion,
  storeData,
};
