
import {z} from "jamfile:zod";

/**
 * Zod schema of https://wttr.in?format=J1,
 * generated with https://transform.tools/json-to-zod.
 *  */ 
export default z.object({
  current_condition: z.array(
    z.object({
      FeelsLikeC: z.string(),
      FeelsLikeF: z.string(),
      cloudcover: z.string(),
      humidity: z.string(),
      localObsDateTime: z.string(),
      observation_time: z.string(),
      precipInches: z.string(),
      precipMM: z.string(),
      pressure: z.string(),
      pressureInches: z.string(),
      temp_C: z.string(),
      temp_F: z.string(),
      uvIndex: z.string(),
      visibility: z.string(),
      visibilityMiles: z.string(),
      weatherCode: z.string(),
      weatherDesc: z.array(z.object({ value: z.string() })),
      weatherIconUrl: z.array(z.object({ value: z.string() })),
      winddir16Point: z.string(),
      winddirDegree: z.string(),
      windspeedKmph: z.string(),
      windspeedMiles: z.string()
    })
  ),
  nearest_area: z.array(
    z.object({
      areaName: z.array(z.object({ value: z.string() })),
      country: z.array(z.object({ value: z.string() })),
      latitude: z.string(),
      longitude: z.string(),
      population: z.string(),
      region: z.array(z.object({ value: z.string() })),
      weatherUrl: z.array(z.object({ value: z.string() }))
    })
  ),
  request: z.array(z.object({ query: z.string(), type: z.string() })),
  weather: z.array(
    z.object({
      astronomy: z.array(
        z.object({
          moon_illumination: z.string(),
          moon_phase: z.string(),
          moonrise: z.string(),
          moonset: z.string(),
          sunrise: z.string(),
          sunset: z.string()
        })
      ),
      avgtempC: z.string(),
      avgtempF: z.string(),
      date: z.string(),
      hourly: z.array(
        z.object({
          DewPointC: z.string(),
          DewPointF: z.string(),
          FeelsLikeC: z.string(),
          FeelsLikeF: z.string(),
          HeatIndexC: z.string(),
          HeatIndexF: z.string(),
          WindChillC: z.string(),
          WindChillF: z.string(),
          WindGustKmph: z.string(),
          WindGustMiles: z.string(),
          chanceoffog: z.string(),
          chanceoffrost: z.string(),
          chanceofhightemp: z.string(),
          chanceofovercast: z.string(),
          chanceofrain: z.string(),
          chanceofremdry: z.string(),
          chanceofsnow: z.string(),
          chanceofsunshine: z.string(),
          chanceofthunder: z.string(),
          chanceofwindy: z.string(),
          cloudcover: z.string(),
          diffRad: z.string(),
          humidity: z.string(),
          precipInches: z.string(),
          precipMM: z.string(),
          pressure: z.string(),
          pressureInches: z.string(),
          shortRad: z.string(),
          tempC: z.string(),
          tempF: z.string(),
          time: z.string(),
          uvIndex: z.string(),
          visibility: z.string(),
          visibilityMiles: z.string(),
          weatherCode: z.string(),
          weatherDesc: z.array(z.object({ value: z.string() })),
          weatherIconUrl: z.array(z.object({ value: z.string() })),
          winddir16Point: z.string(),
          winddirDegree: z.string(),
          windspeedKmph: z.string(),
          windspeedMiles: z.string()
        })
      ),
      maxtempC: z.string(),
      maxtempF: z.string(),
      mintempC: z.string(),
      mintempF: z.string(),
      sunHour: z.string(),
      totalSnow_cm: z.string(),
      uvIndex: z.string()
    })
  )
});
