package com.example;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonArray;
import org.apache.http.HttpHeaders;
import org.apache.http.HttpResponse;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;
import java.util.Base64;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class GenerateTextFromTextInput {
    private static final String GEMINI_API_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent";
    private static final String GEMINI_VISION_API_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent";
    private static final String GEMINI_CHAT_API_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent";
    private static final String API_KEY = System.getenv("GEMINI_API_KEY");

    public static void main(String[] args) {
        if (API_KEY == null) {
            System.err.println("Error: GEMINI_API_KEY environment variable is not set");
            return;
        }

        try (CloseableHttpClient client = HttpClients.createDefault()) {
            // Basic text generation with system instructions
            System.out.println("\n1. Text Generation with System Instructions:");
            generateWithSystemInstructions(client);

            // Multimodal input (assuming you have an image file)
            System.out.println("\n2. Multimodal Input (Vision):");
            // analyzeImage(client, "path/to/your/image.jpg");

            // Chat conversation
            System.out.println("\n3. Chat Conversation:");
            chatConversation(client);

        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static void generateWithSystemInstructions(CloseableHttpClient client) throws Exception {
        String requestBody = """
            {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": "You are a helpful AI assistant. Please explain how AI works in a few words."
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.7,
                    "topP": 0.95,
                    "topK": 40,
                    "maxOutputTokens": 100
                }
            }""";

        executeRequest(client, GEMINI_API_ENDPOINT, requestBody);
    }

    private static void analyzeImage(CloseableHttpClient client, String imagePath) throws Exception {
        // Read and encode image file
        byte[] imageBytes = Files.readAllBytes(Paths.get(imagePath));
        String base64Image = Base64.getEncoder().encodeToString(imageBytes);

        String requestBody = String.format("""
            {
                "contents": [
                    {
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "image/jpeg",
                                    "data": "%s"
                                }
                            },
                            {
                                "text": "What do you see in this image?"
                            }
                        ]
                    }
                ]
            }""", base64Image);

        executeRequest(client, GEMINI_VISION_API_ENDPOINT, requestBody);
    }

    private static void chatConversation(CloseableHttpClient client) throws Exception {
        String requestBody = """
            {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": "Hello"
                            }
                        ]
                    },
                    {
                        "role": "model",
                        "parts": [
                            {
                                "text": "Hi there! How can I help you today?"
                            }
                        ]
                    },
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": "I have two dogs in my house. How many paws are in my house?"
                            }
                        ]
                    }
                ]
            }""";

        executeRequest(client, GEMINI_CHAT_API_ENDPOINT, requestBody);
    }

    private static void executeRequest(CloseableHttpClient client, String endpoint, String requestBody) throws Exception {
        HttpPost request = new HttpPost(endpoint + "?key=" + API_KEY);
        request.setHeader(HttpHeaders.CONTENT_TYPE, "application/json");
        request.setEntity(new StringEntity(requestBody));

        HttpResponse response = client.execute(request);
        String responseBody = EntityUtils.toString(response.getEntity());
        System.out.println("Raw response: " + responseBody);

        JsonObject jsonResponse = new Gson().fromJson(responseBody, JsonObject.class);
        if (jsonResponse.has("candidates")) {
            String generatedText = jsonResponse
                .getAsJsonArray("candidates")
                .get(0)
                .getAsJsonObject()
                .getAsJsonArray("content")
                .get(0)
                .getAsJsonObject()
                .getAsJsonArray("parts")
                .get(0)
                .getAsJsonObject()
                .get("text")
                .getAsString();

            System.out.println("\nAI Response: " + generatedText);
        } else if (jsonResponse.has("error")) {
            System.err.println("API Error: " + jsonResponse.getAsJsonObject("error").get("message").getAsString());
        }
    }
}