import {genkit, z} from "genkit";
import {vertexAI} from "@genkit-ai/vertexai";
import path from 'path';
import {readFile} from "fs/promises";
import {PDFParse} from 'pdf-parse';
import {chunk} from 'llm-chunk';
import {Document} from 'genkit/retriever';
import {enableFirebaseTelemetry} from "@genkit-ai/firebase";
import {initializeApp} from "firebase-admin/app";
import {getFirestore, DocumentData, FieldValue} from 'firebase-admin/firestore';
import {defineFirestoreRetriever} from '@genkit-ai/firebase';


// DEV only
import dotenv from 'dotenv'

dotenv.config({
    path: path.resolve(__dirname, '../../.env')
});


const app = initializeApp();
let firestore = getFirestore(app);

// if (process.env.GOOGLE_APPLICATION_CREDENTIALS) {
//     const dataBuffer = await readFile(process.env.GOOGLE_APPLICATION_CREDENTIALS);
//     const serviceAccountCreds = JSON.parse(process.env.GOOGLE_APPLICATION_CREDENTIALS);
//     console.log(serviceAccountCreds);
//     const authOptions = { credentials: serviceAccountCreds };
//     firestore.settings(authOptions);
// }


// import { defineSecret } from "firebase-functions/params";
// const apiKey = defineSecret("GOOGLE_GENAI_API_KEY");


// const apiKey = defineSecret("GEMINI_API_KEY");


const location = process.env.GCLOUD_LOCATION || '';


enableFirebaseTelemetry();


const collectionName = "hr_documents";

const ai = genkit({
    plugins: [
        vertexAI({location: location,}),
    ],

});


const ragFlow = ai.defineFlow({
        name: "ragFlow",
        outputSchema: z.string(),
    }, async (subject, {sendChunk}) => {


        // Load file content
        const pdfText = await ai.run("extraction", async (): Promise<string> => {
            const pdfFile = path.resolve('./assets/HR-Policies-Manuals.pdf');
            const dataBuffer = await readFile(pdfFile);
            const pdfData = new PDFParse({data: dataBuffer});
            const textResult = await pdfData.getText() || '';
            console.log('PDF text extracted, length: ' + textResult.text.length);
            return textResult.text;
        });

        console.log('===> File loaded Done');


        // Chunking using Split method and create a Document List
        const chunksList: Document[] = await ai.run("chunking", async () => {

            const chunkedData = chunk(pdfText, {
                maxLength: 5000,
                minLength: 4000,
                splitter: 'sentence',
                overlap: 100,
                delimiters: '',
            });
            console.log('Chunked content length ', chunkedData.length);

            return chunkedData;
        })



        const documentsList: Document[] = await ai.run("embeddings", async () => {

          let documents = [];

            for (let i = 0; i < chunksList.length; i++) {
                console.log(i);

                const embedding = (await ai.embed({
                    embedder: vertexAI.embedder('text-embedding-005'),
                    content: chunksList[i],
                }))[0].embedding;

                await new Promise((resolve) => setTimeout(resolve, 10000));


                const document: DocumentData = {
                    content: chunksList[i],
                    embedding: FieldValue.vector(embedding),
                    timeStamp: FieldValue.serverTimestamp()
                };
                documents.push(document);

            }

            return documents;
        })


        // Update Database
        const batch = firestore.batch();
        documentsList.forEach((doc) => {
            const ref = firestore.collection(collectionName).doc();
            batch.create(ref, doc);
        });
        await batch.commit();

        console.log('===> Database Updates');


        // Run CLI
        console.log('===> Indexing Done');


        return "Indexing Done";
    }
);

// Retrieval 
const retrieveResult = ai.defineFlow({
    name: "retrieveResult",
    inputSchema: z.string(),
    outputSchema: z.string(),
}, async input => {

    const firestoreQueryRetriever = defineFirestoreRetriever(ai, {
        name: 'firestoreQueryRetriever',
        firestore,
        collection: collectionName,
        contentField: 'content', // Field containing document content
        vectorField: 'embedding', // Field containing vector embeddings
        embedder: vertexAI.embedder('text-embedding-005'), // Embedder to generate embeddings
        distanceMeasure: 'COSINE', // Default is 'COSINE'; other options: 'EUCLIDEAN', 'DOT_PRODUCT'
    });

    const docs = await ai.retrieve({
        retriever:firestoreQueryRetriever,
        query: input.toString(),
        options: {
            limit: 2,
        },
    });

    const resultAsJson = docs.map((doc, i) => ({
        page: i + 1,
        text: doc.text
    }));

    console.log(JSON.stringify(resultAsJson, null, 2));


    // Generate response using retrieved results
    const prompt =
        `
        Be a helpful assistant and Answer user query in simple as informative , 
        short and only relevant answer for user, 

        \n User Query: ${input.toString()} \n Context ${JSON.stringify(resultAsJson, null, 2)}
        `;
         console.log(prompt);

        
    const { response  } = ai.generateStream({
        model: vertexAI.model('gemini-2.5-pro'),
        prompt: prompt,
        config: {
            temperature: 1,
        },
    });



    return (await response).text;


});

// export const menuSuggestion = onCallGenkit({
//   secrets: [apiKey],
// }, menuSuggestionFlow);
